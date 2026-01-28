import argparse
import json
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import matplotlib.patches as patches


def units_per_cell_from_arrays(
    unit_positions: np.ndarray,
    num_juris: int,
    per_juris_rows: int,
    per_juris_cols: int,
) -> np.ndarray:
    """
    Vectorized counts of units at each cell, ignoring in-transit (cell < 0).
    Returns (J,R,C) int16.
    """
    units_per_cell = np.zeros((num_juris, per_juris_rows, per_juris_cols), dtype=np.int16)

    loc_j = unit_positions[:, 0].astype(int, copy=False)
    cell = unit_positions[:, 1].astype(int, copy=False)
    mask = cell >= 0
    if not np.any(mask):
        return units_per_cell

    cell = cell[mask]
    loc_j = loc_j[mask]
    r = cell // per_juris_cols
    c = cell % per_juris_cols
    np.add.at(units_per_cell, (loc_j, r, c), 1)
    return units_per_cell


def default_tile_positions(
    num_juris: int,
    num_juris_rows: int | None = None,
    num_juris_cols: int | None = None,
) -> np.ndarray:
    if (
        num_juris_rows is not None
        and num_juris_cols is not None
        and num_juris_rows * num_juris_cols == num_juris
    ):
        pos = np.zeros((num_juris, 2), dtype=float)
        for j in range(num_juris):
            r = j // num_juris_cols
            c = j % num_juris_cols
            pos[j] = (c, -r)
        return pos

    side = int(math.ceil(math.sqrt(num_juris)))
    pos = np.zeros((num_juris, 2), dtype=float)
    for j in range(num_juris):
        r = j // side
        c = j % side
        pos[j] = (c, -r)
    return pos


def compute_tile_extents(
    num_juris: int,
    rows: int,
    cols: int,
    tile_pad: int | None = None,
    num_juris_rows: int | None = None,
    num_juris_cols: int | None = None,
):
    tile_w = cols
    tile_h = rows
    if tile_pad is None:
        tile_pad = max(2, int(0.25 * max(rows, cols)))

    base_pos = default_tile_positions(num_juris, num_juris_rows=num_juris_rows, num_juris_cols=num_juris_cols)
    canvas_pos = np.zeros_like(base_pos)
    canvas_pos[:, 0] = base_pos[:, 0] * (tile_w + tile_pad)
    canvas_pos[:, 1] = base_pos[:, 1] * (tile_h + tile_pad)

    extents_tiles = []
    for j in range(num_juris):
        x0 = float(canvas_pos[j, 0])
        y_top = float(canvas_pos[j, 1])
        tile_extent = (x0, x0 + tile_w, y_top - tile_h, y_top)
        extents_tiles.append(tile_extent)
    return extents_tiles


def load_snapshot_files(snapshot_dir: str) -> list[Path]:
    base = Path(snapshot_dir)
    if not base.is_dir():
        raise ValueError(f"Snapshot directory not found: {snapshot_dir}")
    files = sorted(base.glob("*.npz"))
    if not files:
        raise ValueError(f"No snapshot files found in {snapshot_dir}")
    return files


def _meta_path_for_snapshot(snapshot_file: Path) -> Path:
    return snapshot_file.with_name(f"{snapshot_file.stem}__meta.json")


def _adj_matrix_from_coordinates(coordinates_of_each_juris: list[list[float]] | list[tuple[float, float]]) -> np.ndarray:
    coordinates_of_each_juris = [list(p) for p in coordinates_of_each_juris]
    num_juris = len(coordinates_of_each_juris)
    adj_matrix = np.zeros((num_juris, num_juris), dtype=int)
    for i in range(num_juris):
        for j in range(num_juris):
            dist = math.sqrt(
                (coordinates_of_each_juris[i][0] - coordinates_of_each_juris[j][0]) ** 2
                + (coordinates_of_each_juris[i][1] - coordinates_of_each_juris[j][1]) ** 2
            )
            adj_matrix[i, j] = int(math.ceil(dist))
    return adj_matrix


def _adj_matrix_from_mesh(num_juris_rows: int, num_juris_cols: int, juris_travel_time: int) -> np.ndarray:
    if num_juris_rows <= 0 or num_juris_cols <= 0:
        raise ValueError("num_juris_rows and num_juris_cols must be positive.")
    if juris_travel_time < 0:
        raise ValueError("juris_travel_time must be >= 0.")

    num_juris = num_juris_rows * num_juris_cols
    adj_matrix = np.full((num_juris, num_juris), -1, dtype=int)
    for j in range(num_juris):
        adj_matrix[j, j] = 0
        r = j // num_juris_cols
        c = j % num_juris_cols
        if r > 0:
            adj_matrix[j, j - num_juris_cols] = juris_travel_time
        if r < num_juris_rows - 1:
            adj_matrix[j, j + num_juris_cols] = juris_travel_time
        if c > 0:
            adj_matrix[j, j - 1] = juris_travel_time
        if c < num_juris_cols - 1:
            adj_matrix[j, j + 1] = juris_travel_time
    return adj_matrix


def load_snapshot_metadata(snapshot_file: Path) -> dict:
    meta_path = _meta_path_for_snapshot(snapshot_file)
    if not meta_path.is_file():
        raise ValueError(
            f"Metadata file not found for {snapshot_file.name}. Expected {meta_path.name}. "
            "Re-run the simulator with --save-snapshots to generate metadata."
        )
    with open(meta_path, "r", encoding="utf-8") as file:
        return json.load(file)


def animate_snapshot_file(snapshot_file: Path, out_file: Path, fps: float = 2.0):
    data = np.load(snapshot_file)
    burning_map = data["burning_map"]
    if "unit_positions" in data:
        unit_positions = data["unit_positions"]
    else:
        unit_in_which_juris = data["unit_in_which_juris"]
        unit_in_which_cell = data["unit_in_which_cell"]
        loc_j = unit_in_which_juris.reshape(unit_in_which_juris.shape[0], -1)
        cell = unit_in_which_cell.reshape(unit_in_which_cell.shape[0], -1)
        unit_positions = np.stack([loc_j, cell], axis=2)

    steps = burning_map.shape[0] - 1
    J, R, C = burning_map.shape[1:4]

    meta = load_snapshot_metadata(snapshot_file)
    num_juris_rows = meta.get("num_juris_rows")
    num_juris_cols = meta.get("num_juris_cols")
    ext_tiles = compute_tile_extents(
        J,
        R,
        C,
        num_juris_rows=num_juris_rows,
        num_juris_cols=num_juris_cols,
    )
    cmap = mcolors.ListedColormap([mcolors.to_rgb("green"), mcolors.to_rgb("red")])

    if "adj_matrix" in meta:
        adj = np.array(meta["adj_matrix"], dtype=int)
    elif "num_juris_rows" in meta and "num_juris_cols" in meta and "juris_travel_time" in meta:
        adj = _adj_matrix_from_mesh(
            int(meta["num_juris_rows"]),
            int(meta["num_juris_cols"]),
            int(meta["juris_travel_time"]),
        )
    elif "coordinates_of_each_juris" in meta:
        adj = _adj_matrix_from_coordinates(meta["coordinates_of_each_juris"])
    else:
        raise ValueError("Metadata missing adj_matrix, mesh settings, or coordinates_of_each_juris.")

    dest_rows = [
        [k for k in range(J) if k != j and adj[j, k] > 0]
        for j in range(J)
    ]
    dest_row_index = [
        {k: i for i, k in enumerate(dest_rows[j])}
        for j in range(J)
    ]

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    xs = [e[0] for e in ext_tiles] + [e[1] for e in ext_tiles]
    ys = [e[2] for e in ext_tiles] + [e[3] for e in ext_tiles]
    for j in range(J):
        x0, x1, y0, y1 = ext_tiles[j]
        max_len = int(np.max(adj[j])) if J > 0 else 0
        xs.append(x0 + max_len)
        ys.append(y0 - len(dest_rows[j]))
    pad = 4
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    travel_cell_outlines = []
    for j in range(J):
        x0, x1, y0, y1 = ext_tiles[j]
        for row_idx, k in enumerate(dest_rows[j]):
            row_len = int(adj[j, k])
            for t in range(row_len):
                rect = patches.Rectangle(
                    (x0 + t, y0 - (row_idx + 1)),
                    1,
                    1,
                    fill=False,
                    linewidth=0.5,
                    edgecolor="lightgray",
                    alpha=0.7,
                    zorder=1,
                )
                ax.add_patch(rect)
                travel_cell_outlines.append(rect)

    tile_ims = []
    tile_boxes = []
    tile_titles = []
    for j in range(J):
        state = burning_map[0, j].astype(int)
        im = ax.imshow(state, cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
                       extent=ext_tiles[j], origin="upper")
        tile_ims.append(im)

        x0, x1, y0, y1 = ext_tiles[j]
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linewidth=2.0)
        ax.add_patch(rect)
        tile_boxes.append(rect)

        tx = (x0 + x1) / 2
        ty = y1 + 0.8
        title = ax.text(tx, ty, f"J{j}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        tile_titles.append(title)

    unit_patches = []
    unit_texts = []
    transit_patches = []
    transit_texts = []
    transit_info: dict[int, tuple[int, int, int]] = {}
    last_positions = unit_positions[0].copy()

    def clear_units():
        for p in unit_patches:
            p.remove()
        unit_patches.clear()
        for t in unit_texts:
            t.remove()
        unit_texts.clear()
        for p in transit_patches:
            p.remove()
        transit_patches.clear()
        for t in transit_texts:
            t.remove()
        transit_texts.clear()

    def draw_units(frame_idx: int):
        upc = units_per_cell_from_arrays(
            unit_positions[frame_idx],
            num_juris=J,
            per_juris_rows=R,
            per_juris_cols=C,
        )
        for j in range(J):
            x0, x1, y0, y1 = ext_tiles[j]
            cells = np.argwhere(upc[j] > 0)
            for (r, c) in cells:
                cnt = int(upc[j, r, c])
                cx = x0 + c + 0.5
                cy = y1 - (r + 0.5)

                rect = patches.Rectangle((x0 + c, y1 - (r + 1)), 1, 1,
                                         linewidth=1.5, edgecolor="cyan",
                                         facecolor="deepskyblue", alpha=0.7, zorder=5)
                ax.add_patch(rect)
                unit_patches.append(rect)

                txt = ax.text(cx, cy, str(cnt), ha="center", va="center",
                              color="white", fontsize=8, fontweight="bold", zorder=6)
                unit_texts.append(txt)

    def update(frame_idx: int):
        nonlocal last_positions
        for j in range(J):
            tile_ims[j].set_data(burning_map[frame_idx, j].astype(int))

        clear_units()
        draw_units(frame_idx)

        cur_positions = unit_positions[frame_idx]
        if frame_idx > 0:
            prev_cells = last_positions[:, 1]
            prev_juris = last_positions[:, 0]
            cur_cells = cur_positions[:, 1]
            cur_juris = cur_positions[:, 0]

            starting = (prev_cells >= 0) & (cur_cells < 0)
            for idx in np.nonzero(starting)[0]:
                from_j = int(prev_juris[idx])
                to_j = int(cur_juris[idx])
                total_steps = int(adj[from_j, to_j])
                transit_info[idx] = (from_j, to_j, total_steps)

            arrived = (cur_cells >= 0)
            for idx in np.nonzero(arrived)[0]:
                transit_info.pop(int(idx), None)

        transit_counts: dict[tuple[int, int, int], int] = {}
        for unit_idx in range(cur_positions.shape[0]):
            cell = int(cur_positions[unit_idx, 1])
            if cell >= 0:
                continue

            info = transit_info.get(unit_idx)
            if info is None:
                continue

            from_j, to_j, total_steps = info
            total_steps = max(total_steps, 1)
            remaining = -cell
            pos = max(0, min(total_steps - 1, total_steps - remaining))

            row_idx = dest_row_index[from_j].get(to_j, None)
            if row_idx is None:
                continue

            key = (from_j, row_idx, pos)
            transit_counts[key] = transit_counts.get(key, 0) + 1

        for (from_j, row_idx, pos), cnt in transit_counts.items():
            x0, x1, y0, y1 = ext_tiles[from_j]
            cx = x0 + pos + 0.5
            cy = y0 - (row_idx + 0.5)

            rect = patches.Rectangle(
                (x0 + pos, y0 - (row_idx + 1)),
                1,
                1,
                linewidth=1.0,
                edgecolor="cyan",
                facecolor="deepskyblue",
                alpha=0.7,
                zorder=5,
            )
            ax.add_patch(rect)
            transit_patches.append(rect)

            txt = ax.text(cx, cy, str(cnt), ha="center", va="center",
                          color="white", fontsize=7, fontweight="bold", zorder=6)
            transit_texts.append(txt)

        last_positions = cur_positions.copy()

        total_burning = int(np.sum(burning_map[frame_idx]))
        total_in_transit = int(np.sum(unit_positions[frame_idx, :, 1] < 0))
        ax.set_title(
            f"Fire Simulation | Step {frame_idx:03d}\n"
            f"Burning total: {total_burning} | In-transit units: {total_in_transit}",
            fontsize=12
        )

        artists = tile_ims + tile_boxes + tile_titles + unit_patches + unit_texts
        artists += transit_patches + transit_texts
        return artists

    anim = animation.FuncAnimation(fig, update, frames=steps + 1, interval=int(1000 / fps), blit=True, repeat=True)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving animation to {out_file} ...")
    try:
        anim.save(str(out_file), writer="pillow", fps=fps, savefig_kwargs={"bbox_inches": "tight"})
        print("Saved with pillow.")
    except Exception as e:
        print(f"pillow save failed: {e}")
        mp4 = out_file.with_suffix(".mp4")
        try:
            anim.save(str(mp4), writer="ffmpeg", fps=fps, savefig_kwargs={"bbox_inches": "tight"})
            print(f"Saved as {mp4} with ffmpeg.")
        except Exception as e2:
            print(f"ffmpeg save failed too: {e2}")
            print("Try: pip install pillow  OR  install ffmpeg")

    plt.close(fig)
    return anim


def main():
    parser = argparse.ArgumentParser(description="Animate from saved snapshot files.")
    parser.add_argument("--snapshots-dir", default="snapshots", help="Directory containing .npz snapshot files.")
    parser.add_argument("--output-dir", default="animations", help="Directory for output animations.")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second.")
    args = parser.parse_args()

    files = load_snapshot_files(args.snapshots_dir)
    out_dir = Path(args.output_dir)

    for snapshot_file in files:
        out_file = out_dir / f"{snapshot_file.stem}.gif"
        animate_snapshot_file(snapshot_file, out_file, fps=args.fps)


if __name__ == "__main__":
    main()
