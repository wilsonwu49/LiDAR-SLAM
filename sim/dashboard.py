"""Step-by-step dashboard of a simulated run: one panel per stage of the pipeline.

    python -m sim.dashboard --world maze                    MP4 in docs/videos/
    python -m sim.dashboard --world crowd --live            watch it in a window
    python -m sim.dashboard --world maze --live --mp4 maze.mp4
    python -m sim.dashboard --world clutter --every 3 --speed 0.5

worlds: doorway, clutter, crowd, maze.   Live window: space pauses, q quits.
"""

import argparse
import os
import time

import numpy as np
import matplotlib

import lidar_nav.navigate as nav
import sim.scenarios as sc
from lidar_nav.control.controller import ControlParams
from sim.sensors import SimSensors
from sim.worlds import WORLDS

W_PX, H_PX, DPI = 1632, 1088, 100
GREEN, BLUE, ORANGE, RED, GREY = "#2e8b57", "#3a6ea5", "#f18f01", "#d1495b", "#7a828e"
ZOOM = 0.9


def unicycle_arc(pose, u, horizon=1.5, n=20):
    x, y, th = pose
    pts = [(x, y)]
    dt = horizon / n
    for _ in range(n):
        x += u[0] * np.cos(th) * dt
        y += u[0] * np.sin(th) * dt
        th += u[1] * dt
        pts.append((x, y))
    return np.array(pts)


def ellipse_2sigma(P2):
    w, V = np.linalg.eigh(P2)
    w = np.maximum(w, 0.0)
    return 4 * np.sqrt(w[1]), 4 * np.sqrt(w[0]), np.degrees(np.arctan2(V[1, 1], V[0, 1]))


class Dashboard:
    """Six panels, built once and updated in place each frame."""

    def __init__(self, world, params, grid, live):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Ellipse

        self.plt, self.Circle = plt, Circle
        self.world, self.params, self.grid = world, params, grid
        self.paused = False
        self.closed = False
        fig, axs = plt.subplots(2, 3, figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI)
        self.fig, self.ax = fig, axs.ravel()
        fig.subplots_adjust(left=0.03, right=0.995, top=0.895, bottom=0.035, wspace=0.12, hspace=0.2)
        seg = world["segments"]
        pad = 0.3
        self.xlim = (seg[:, [0, 2]].min() - pad, seg[:, [0, 2]].max() + pad)
        self.ylim = (seg[:, [1, 3]].min() - pad, seg[:, [1, 3]].max() + pad)
        extent = (grid.origin_x, grid.origin_x + grid.W * grid.s,
                  grid.origin_y, grid.origin_y + grid.H * grid.s)
        titles = ["1  SENSE: raw LiDAR scan, robot's own view",
                  "2  ESTIMATE: where it thinks it is (zoomed)",
                  "3  MAP: what it has built from scans",
                  "4  PLAN: A* route on the inflated map",
                  "5  SAFETY FILTER: command before and after",
                  "6  REAL WORLD: what is actually happening"]
        for a, t in zip(self.ax, titles):
            a.set_title(t, loc="left", fontsize=10, fontweight="bold")
            a.set_aspect("equal")
            a.tick_params(labelsize=7)
        for a in self.ax[[2, 3, 5]]:
            a.set_xlim(*self.xlim)
            a.set_ylim(*self.ylim)

        a = self.ax[0]
        zm = world["z_max"]
        view = min(zm, 0.55 * max(self.xlim[1] - self.xlim[0], self.ylim[1] - self.ylim[0]))
        a.set_xlim(-view, view)
        a.set_ylim(-view, view)
        a.add_patch(Circle((0, 0), zm, fill=False, ls=":", color=GREY))
        a.plot([0, 0.6], [0, 0], color="k", lw=2)
        a.add_patch(Circle((0, 0), params.r_robot, color="k"))
        self.scan_pts = a.scatter([], [], s=3, color=RED)
        self.sense_txt = a.text(0.02, 0.02, "", transform=a.transAxes, fontsize=8, family="monospace",
                                va="bottom", bbox=dict(fc="white", ec="none", alpha=0.85))
        a.set_xlabel("forward (m)", fontsize=8)

        a = self.ax[1]
        for x0, y0, x1, y1 in seg:
            a.plot([x0, x1], [y0, y1], color="#bbb", lw=2, zorder=0)
        self.true_trail_z, = a.plot([], [], color=GREEN, lw=1.5)
        self.est_trail_z, = a.plot([], [], color=BLUE, lw=1.5, ls="--")
        self.ell = Ellipse((0, 0), 0, 0, facecolor=BLUE, alpha=0.25, edgecolor=BLUE, zorder=4)
        a.add_patch(self.ell)
        self.true_body = Circle((0, 0), params.r_robot, color=GREEN, alpha=0.8, zorder=5)
        self.est_body = Circle((0, 0), params.r_robot, fill=False, edgecolor=BLUE, lw=2, zorder=6)
        a.add_patch(self.true_body)
        a.add_patch(self.est_body)
        self.true_head, = a.plot([], [], color="white", lw=2, zorder=7)
        self.est_head, = a.plot([], [], color=BLUE, lw=2, zorder=7)
        self.est_txt = a.text(0.02, 0.02, "", transform=a.transAxes, fontsize=8, family="monospace",
                              va="bottom", bbox=dict(fc="white", ec="none", alpha=0.85))

        a = self.ax[2]
        self.map_img = a.imshow(np.full((grid.H, grid.W), 128, np.uint8), origin="lower", extent=extent,
                                cmap="gray_r", vmin=0, vmax=255, interpolation="nearest")
        self.map_trail, = a.plot([], [], color=BLUE, lw=1.3)
        self.map_robot = Circle((0, 0), params.r_robot, color=BLUE, zorder=5)
        a.add_patch(self.map_robot)

        a = self.ax[3]
        self.plan_img = a.imshow(np.ones((grid.H, grid.W, 3)), origin="lower", extent=extent,
                                 interpolation="nearest")
        self.plan_line, = a.plot([], [], color=ORANGE, lw=2)
        self.plan_robot = Circle((0, 0), params.r_robot, color=BLUE, zorder=5)
        a.add_patch(self.plan_robot)
        a.plot(*world["goal"], marker="*", color="#e0a100", ms=16, zorder=6)
        self.plan_txt = a.text(0.02, 0.02, "", transform=a.transAxes, fontsize=8, family="monospace",
                               va="bottom", bbox=dict(fc="white", ec="none", alpha=0.85))

        a = self.ax[4]
        self.obs_pts = a.scatter([], [], s=28, marker="x", color=RED, zorder=4)
        self.safety_ring = Circle((0, 0), params.R, fill=False, ls="--", lw=1.5, edgecolor=GREY, zorder=3)
        a.add_patch(self.safety_ring)
        self.f_body = Circle((0, 0), params.r_robot, color=BLUE, alpha=0.9, zorder=5)
        a.add_patch(self.f_body)
        self.nom_arc, = a.plot([], [], color=ORANGE, lw=2.2, ls="--", label="pure pursuit wants")
        self.out_arc, = a.plot([], [], color=GREEN, lw=2.6, label="filter sends")
        self.f_path, = a.plot([], [], color="#ddd", lw=6, zorder=0)
        a.legend(loc="upper right", fontsize=7, frameon=False)
        self.filt_txt = a.text(0.02, 0.02, "", transform=a.transAxes, fontsize=8, family="monospace",
                               va="bottom", bbox=dict(fc="white", ec="none", alpha=0.85))

        a = self.ax[5]
        for x0, y0, x1, y1 in seg:
            a.plot([x0, x1], [y0, y1], color="#222", lw=2.2)
        self.mover_patches = [Circle((0, 0), m["radius"], color=RED, alpha=0.85, zorder=4)
                              for m in world["movers"]]
        for c in self.mover_patches:
            a.add_patch(c)
        self.world_trail, = a.plot([], [], color=GREEN, lw=1.5)
        self.world_robot = Circle((0, 0), params.r_robot, color=GREEN, zorder=5)
        a.add_patch(self.world_robot)
        self.world_head, = a.plot([], [], color="white", lw=2, zorder=6)
        a.plot(*world["goal"], marker="*", color="#e0a100", ms=16, zorder=6)

        self.header = fig.text(0.03, 0.965, "", fontsize=13, fontweight="bold", va="center")
        self.events = fig.text(0.03, 0.932, "", fontsize=10.5, va="center", color="#333")

        if live:
            plt.ion()
            fig.canvas.mpl_connect("key_press_event", self._key)
            fig.canvas.mpl_connect("close_event", lambda e: setattr(self, "closed", True))
            plt.show(block=False)

    def _key(self, e):
        if e.key == " ":
            self.paused = not self.paused
        elif e.key == "q":
            self.closed = True

    def update(self, f):
        p = self.params
        tp, ep = f["true_pose"], f["est_pose"]

        a = f["angles"]
        r = f["ranges"]
        ok = np.isfinite(r)
        self.scan_pts.set_offsets(np.column_stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])]))
        self.scan_pts.set_color(RED if f["new_scan"] else "#e9a3ad")
        icp = {None: "not run", True: "accepted", False: "rejected"}[f["icp_ok"]]
        self.sense_txt.set_text(
            f"{'NEW SCAN' if f['new_scan'] else 'same scan'}  {ok.sum()} returns\n"
            f"encoder  {f['encoder'][0]:+.2f} m/s\n"
            f"gyro     {np.degrees(f['w_m'][2]):+.1f} deg/s\n"
            f"ICP      {icp}")

        tt = f["true_trail"]
        et = f["est_trail"]
        self.true_trail_z.set_data(tt[:, 0], tt[:, 1])
        self.est_trail_z.set_data(et[:, 0], et[:, 1])
        w, h, ang = ellipse_2sigma(f["P3"][:2, :2])
        self.ell.set_center(ep[:2])
        self.ell.width, self.ell.height, self.ell.angle = w, h, ang
        self.true_body.set_center(tp[:2])
        self.est_body.set_center(ep[:2])
        hl = 1.6 * p.r_robot
        self.true_head.set_data([tp[0], tp[0] + hl * np.cos(tp[2])], [tp[1], tp[1] + hl * np.sin(tp[2])])
        self.est_head.set_data([ep[0], ep[0] + hl * np.cos(ep[2])], [ep[1], ep[1] + hl * np.sin(ep[2])])
        self.ax[1].set_xlim(tp[0] - ZOOM, tp[0] + ZOOM)
        self.ax[1].set_ylim(tp[1] - ZOOM, tp[1] + ZOOM)
        err = np.linalg.norm(tp[:2] - ep[:2]) * 1000
        sx, sy = np.sqrt(max(f["P3"][0, 0], 0)) * 1000, np.sqrt(max(f["P3"][1, 1], 0)) * 1000
        self.est_txt.set_text(f"real error  {err:5.0f} mm\nσx {sx:5.1f}  σy {sy:5.1f} mm\n"
                              f"green = truth, blue = belief")

        self.map_img.set_data(f["prob"])
        self.map_trail.set_data(et[:, 0], et[:, 1])
        self.map_robot.set_center(ep[:2])

        gray = 1.0 - f["prob"].astype(float)[..., None] / 255.0
        rgb = np.repeat(gray, 3, axis=2)
        if f["lethal"] is not None:
            tint = f["lethal"] & (f["prob"] < 200)
            rgb[tint] = rgb[tint] * 0.55 + np.array([0.95, 0.55, 0.55]) * 0.45
        self.plan_img.set_data(rgb)
        if f["path"] is not None:
            self.plan_line.set_data(f["path"][:, 0], f["path"][:, 1])
            plen = np.linalg.norm(np.diff(f["path"], axis=0), axis=1).sum()
            self.plan_txt.set_text(f"route {plen:4.1f} m   replans so far {f['replans']}\n"
                                   f"pink = too close to a wall for the robot")
        else:
            self.plan_line.set_data([], [])
            self.plan_txt.set_text(f"NO ROUTE: waiting   replans so far {f['replans']}")
        self.plan_robot.set_center(ep[:2])

        a4 = self.ax[4]
        a4.set_xlim(ep[0] - ZOOM, ep[0] + ZOOM)
        a4.set_ylim(ep[1] - ZOOM, ep[1] + ZOOM)
        self.obs_pts.set_offsets(f["obstacles"] if len(f["obstacles"]) else np.zeros((0, 2)))
        feeler = ep[:2] + p.ell * np.array([np.cos(ep[2]), np.sin(ep[2])])
        self.safety_ring.set_center(feeler)
        self.safety_ring.set_radius(f["R_eff"])
        self.safety_ring.set_edgecolor(RED if f["active"] else GREY)
        self.f_body.set_center(ep[:2])
        if f["path"] is not None:
            self.f_path.set_data(f["path"][:, 0], f["path"][:, 1])
        un, uo = f["u_nom"], f["u"]
        if np.all(np.isfinite(un)):
            arc = unicycle_arc(ep, un)
            self.nom_arc.set_data(arc[:, 0], arc[:, 1])
        else:
            self.nom_arc.set_data([], [])
        arc = unicycle_arc(ep, uo)
        self.out_arc.set_data(arc[:, 0], arc[:, 1])
        hmin = f["h_min"]
        self.filt_txt.set_text(
            f"wants  v {un[0]:+.2f} m/s  ω {un[1]:+.2f} rad/s\n"
            f"sends  v {uo[0]:+.2f} m/s  ω {uo[1]:+.2f} rad/s\n"
            f"{'ACTIVE: changing the command' if f['active'] else 'idle: command passes through'}\n"
            f"obstacles watched: {f['n_obs']}   margin "
            f"{'-' if not np.isfinite(hmin) else f'{hmin * 1000:.0f} mm'}")

        for c, m in zip(self.mover_patches, f["movers"]):
            c.set_center(m)
        self.world_trail.set_data(tt[:, 0], tt[:, 1])
        self.world_robot.set_center(tp[:2])
        self.world_head.set_data([tp[0], tp[0] + hl * np.cos(tp[2])], [tp[1], tp[1] + hl * np.sin(tp[2])])

        self.header.set_text(f"{self.world['name'].upper()}    t = {f['t']:5.1f} s    "
                             f"distance to goal {f['to_goal']:4.1f} m")
        self.events.set_text("   |   ".join(f["events"]) if f["events"] else "driving")

    def rgb(self):
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())[..., :3]
        if buf.shape[:2] != (H_PX, W_PX):
            from PIL import Image
            buf = np.asarray(Image.fromarray(buf).resize((W_PX, H_PX)))
        return buf

    def show(self):
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.plt.pause(0.001)
        while self.paused and not self.closed:
            self.plt.pause(0.05)


class VideoOut:
    """MP4 through imageio's bundled ffmpeg, falling back to numbered PNGs."""

    def __init__(self, path, fps):
        self.path, self.fps, self.n = path, fps, 0
        self.writer, self.png_dir = None, None
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        try:
            import imageio.v2 as imageio
            self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7,
                                             macro_block_size=16, pixelformat="yuv420p")
        except Exception as e:
            self.png_dir = os.path.splitext(path)[0] + "_frames"
            os.makedirs(self.png_dir, exist_ok=True)
            print(f"could not start ffmpeg ({e.__class__.__name__}: {e}); writing PNG frames to {self.png_dir}")

    def add(self, rgb):
        if self.writer is not None:
            self.writer.append_data(rgb)
        else:
            from PIL import Image
            Image.fromarray(rgb).save(os.path.join(self.png_dir, f"frame_{self.n:05d}.png"))
        self.n += 1

    def close(self):
        if self.writer is not None:
            self.writer.close()
            return self.path
        return self.png_dir


def run(world_name, live=False, mp4=None, every=5, speed=1.0, k=0.0, seed=1, slip=0.0,
        prior_map=False, max_cycles=None):
    if not live:
        matplotlib.use("Agg")
    w = WORLDS[world_name]()
    params = ControlParams()
    params.k = k
    grid = sc.make_grid(*w["grid_wh"])
    grid.z_max = w["z_max"]
    if prior_map:
        for x0, y0, x1, y1 in w["segments"]:
            for t in np.linspace(0, 1, int(np.hypot(x1 - x0, y1 - y0) / (grid.s * 0.5)) + 2):
                cx, cy = grid.world_to_cell(x0 + t * (x1 - x0), y0 + t * (y1 - y0))
                if grid.in_bounds(cx, cy):
                    grid.logodds[cy, cx] = 5.0
        grid.integrate_scan = lambda *a, **kw: None
    ekf = sc.make_ekf(w["start"])
    sens = SimSensors(w["segments"], w["start"], params, dt=sc.DT, z_max=w["z_max"], seed=seed,
                      movers=w["movers"], slip_frac=slip)

    dash = Dashboard(w, params, grid, live)
    video = VideoOut(mp4, fps=max(1, round(speed / (sc.DT * every)))) if mp4 else None
    est_trail, pending = [], []
    state = dict(replans=0, last_len=None, t0=time.perf_counter())

    def on_cycle(cycle, pose, P3, path, grid, info, lethal, replanned, sensed):
        est_trail.append(pose[:2].copy())
        if replanned:
            state["replans"] += 1
            if path is not None:
                L = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
                if state["last_len"] is not None and L > state["last_len"] + 1.5:
                    pending.append(f"REPLANNED: route grew {state['last_len']:.1f} → {L:.1f} m "
                                   f"(dead end or blocked)")
                state["last_len"] = L
        if info.get("status") == "waiting" and "WAITING" not in " ".join(pending):
            pending.append("WAITING: no route right now")
        if sensed["new_scan"] and "new scan" not in " ".join(pending):
            pending.append(f"new scan, ICP {'accepted' if sensed['icp_ok'] else 'not used'}")
        if info.get("active") and "safety filter" not in " ".join(pending):
            pending.append("safety filter is changing the command")
        if cycle % every or dash.closed:
            return
        centres, _, _ = sens.mover_state()
        f = dict(
            t=cycle * sc.DT, true_pose=sens.pose_true.copy(), est_pose=pose.copy(), P3=P3.copy(),
            true_trail=np.array(sens.trail)[:, :2], est_trail=np.array(est_trail),
            angles=sensed["angles"], ranges=sensed["ranges"], new_scan=sensed["new_scan"],
            icp_ok=sensed["icp_ok"], encoder=sensed["encoder"], w_m=sensed["w_m"],
            prob=(255.0 / (1.0 + np.exp(-grid.logodds))).astype(np.uint8),
            lethal=None if lethal is None else lethal.copy(),
            path=None if path is None else np.array(path), replans=state["replans"],
            obstacles=np.asarray(info.get("obstacles", np.zeros((0, 2)))),
            n_obs=int(info.get("n_obstacles", 0)), R_eff=float(info.get("R_eff", params.R)),
            active=bool(info.get("active", False)), h_min=float(info.get("h_min", np.nan)),
            u_nom=np.asarray(info.get("u_nom", [np.nan, np.nan]), float),
            u=np.asarray(info.get("u", [0.0, 0.0]), float),
            movers=centres, to_goal=float(np.linalg.norm(sens.pose_true[:2] - np.array(w["goal"]))),
            events=list(pending))
        pending.clear()
        dash.update(f)
        if video:
            video.add(dash.rgb())
        if live:
            dash.show()

    result = nav.navigate(sens, grid, ekf, np.array(w["goal"]), params, dt=sc.DT,
                           max_cycles=max_cycles or w["max_cycles"], use_covariance=k > 0,
                           on_cycle=on_cycle)
    off = np.linalg.norm(sens.pose_true[:2] - np.array(w["goal"]))
    summary = (f"{w['name']}: {result['outcome']} in {result['cycles'] * sc.DT:.1f} s simulated, "
               f"{off * 100:.0f} cm from the goal, {result['replans']} replans, "
               f"{time.perf_counter() - state['t0']:.0f} s to render")
    out = video.close() if video else None
    if live and not dash.closed:
        dash.header.set_text(dash.header.get_text() + f"    {result['outcome'].upper()}")
        dash.plt.ioff()
        dash.plt.show()
    return summary, out


def main_():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default="maze", choices=sorted(WORLDS))
    ap.add_argument("--live", action="store_true", help="show a live window while it runs")
    ap.add_argument("--mp4", default=None, help="video path (default docs/videos/<world>.mp4 unless --live)")
    ap.add_argument("--every", type=int, default=5, help="control cycles per frame (5 = 10 frames per second)")
    ap.add_argument("--speed", type=float, default=1.0, help="video playback speed relative to real time")
    ap.add_argument("--k", type=float, default=0.0, help="uncertainty margin multiplier (0 = fixed radius)")
    ap.add_argument("--slip", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--prior-map", action="store_true")
    a = ap.parse_args()
    mp4 = a.mp4 or (None if a.live else f"docs/videos/{a.world}.mp4")
    summary, out = run(a.world, a.live, mp4, a.every, a.speed, a.k, a.seed, a.slip, a.prior_map)
    print(summary)
    if out:
        print("wrote", out)


if __name__ == "__main__":
    main_()
