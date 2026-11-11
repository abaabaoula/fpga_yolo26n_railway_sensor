"""Publication figures from verified CSVs; no fitted range or extrapolation."""
import csv
import json
from pathlib import Path
import sys
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/railway_distance_1'
OUT = ROOT / 'output/railway_distance_1'
FIG = OUT / 'figures'
sys.path.insert(0, str(Path(__file__).parent))
from analyze_geometry import sha


def rows(name):
    with (REPORT / name).open(newline='', encoding='utf8') as stream:
        return list(csv.DictReader(stream))


def save(fig, name):
    fig.savefig(FIG / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(FIG / (name + '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    reference = rows('reference_linked98.csv')
    detection = rows('detection_by_range.csv')
    depth = rows('depth_instances.csv')
    summary = json.loads((REPORT / 'depth_summary.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={'width_ratios': [1, 1.2]})
    r = np.array([float(v['range_m']) for v in reference])
    axes[0].hist(r, bins=np.arange(120, 191, 10), color='#356aa0', edgecolor='white')
    axes[0].set(xlabel='3D-reference camera-to-centre range (m)', ylabel='Provided person instances',
                xlim=(120, 190), ylim=(0, 44))
    axes[0].text(.03, .96, '98 instances / 20 images / 11 tracks\nCoverage: 98 of 271 external boxes',
                 transform=axes[0].transAxes, va='top', fontsize=9)
    x = np.arange(2)
    styles = [('old_adaptive', 'Custom adaptive', '#356aa0'),
              ('roi_score', 'Region-score policy', '#c16a28'),
              ('ncnn_adaptive', 'NCNN FP32 adaptive', '#747b85')]
    for i, (key, label, color) in enumerate(styles):
        selected = [s for s in detection if s['method'] == key and s['instances'] != '0']
        assert len(selected) == 2
        positions = x+(i-1)*.24
        values = [100*float(s['recall']) for s in selected]
        axes[1].bar(positions, values, width=.23, label=label, color=color)
        for xx, val, s in zip(positions, values, selected):
            axes[1].text(xx, val+.18, f"{s['matched']}/{s['instances']}", ha='center', va='bottom', fontsize=8)
    axes[1].set(xticks=x, xticklabels=['[100, 150) m\n31 instances', '[150, 200) m\n67 instances'],
                ylabel='Provided-box recall (%)', ylim=(0, 10))
    axes[1].legend(loc='upper left', frameon=False, fontsize=8)
    axes[1].text(.97, .96, 'Conf. 0.25; IoU 0.5', transform=axes[1].transAxes,
                 ha='right', va='top', fontsize=8)
    for label, ax in zip('ab', axes):
        ax.text(-.14, 1.06, label, transform=ax.transAxes, fontweight='bold', fontsize=13)
    fig.subplots_adjust(wspace=.36, bottom=.2, top=.90)
    save(fig, 'FigureS2_distance_coverage')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    methods = [('Metric3Dv2_ViT_small', 'Frozen Metric3Dv2 ViT-S', '#356aa0'),
               ('height_prior_1.7m', 'Assumed 1.7 m adult height', '#c16a28')]
    for ax, (method, title, color), label in zip(axes, methods, 'ab'):
        selected = [v for v in depth if v['method'] == method]
        xx = np.array([float(v['reference_z_m']) for v in selected])
        yy = np.array([float(v['estimated_z_m']) for v in selected])
        ax.plot([80, 205], [80, 205], '--', color='#555555', lw=1, label='Identity')
        ax.scatter(xx, yy, s=20, alpha=.65, color=color, edgecolors='none')
        m = summary['methods'][method]
        ax.set(xlabel='3D-reference optical-axis depth (m)', ylabel='Estimated optical-axis depth (m)',
               xlim=(120, 190), ylim=(80, 205), title=title)
        ax.text(.04, .96, f"n = {len(selected)}\nAbsRel = {m['mean_absrel']*100:.1f}%\nMedian absolute error = {m['median_absolute_error_m']:.1f} m",
                transform=ax.transAxes, va='top', fontsize=9)
        ax.text(-.14, 1.1, label, transform=ax.transAxes, fontweight='bold', fontsize=13)
    fig.subplots_adjust(wspace=.30, bottom=.17, top=.85)
    save(fig, 'FigureS3_monocular_validation')

    # A separately licensed local inspection image, excluded from manuscript
    # figures. It illustrates a successful match, not an operating-range claim.
    geometry = json.loads((REPORT / 'geometry_summary.json').read_text())
    success = geometry['methods']['old_adaptive']['farthest_matched_instances'][0]
    farthest = max(reference, key=lambda v: float(v['range_m']))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    provenance = []
    for ax, obj, label, title in zip(axes, [success, farthest], 'ab', ['Observed successful match', 'Farthest referenced instance']):
        image_id = obj['image_id']
        img_path = ROOT / 'output/railway_sensors_validation_1/datasets/OSDaR23_new' / image_id
        rgb = cv2.imread(str(img_path))[:, :, ::-1]
        cx, cy = (float(obj['x0'])+float(obj['x1']))/2, (float(obj['y0'])+float(obj['y1']))/2
        x0, y0 = int(max(0, cx-100)), int(max(0, cy-120))
        x1, y1 = min(rgb.shape[1], x0+200), min(rgb.shape[0], y0+240)
        ax.imshow(rgb[y0:y1, x0:x1])
        ax.add_patch(Rectangle((float(obj['x0'])-x0, float(obj['y0'])-y0),
                               float(obj['x1'])-float(obj['x0']), float(obj['y1'])-float(obj['y0']),
                               fill=False, edgecolor='#ffbf3d', lw=1.4, linestyle='--'))
        if obj is success:
            key = hashlib.sha256(image_id.encode()).hexdigest()[:24]
            c = json.loads((ROOT / 'output/railway_sensors_validation_1/accuracy_external_v1' / (key+'.json')).read_text())
            from railway_person_research.evaluation import iou_matrix
            pred = np.array(c['variants']['old_adaptive']['predictions']).reshape(-1, 6)
            box = [float(obj[k]) for k in ['x0', 'y0', 'x1', 'y1']]
            hits = np.flatnonzero((iou_matrix(pred[:, :4], [box])[:, 0] >= .5) & (pred[:, 4] >= .25))
            pp = pred[hits[0]]
            ax.add_patch(Rectangle((pp[0]-x0, pp[1]-y0), pp[2]-pp[0], pp[3]-pp[1],
                                   fill=False, edgecolor='#27e6b0', lw=1.4))
        ax.set_title(f'{label}  {title}\n3D reference: {float(obj["range_m"]):.1f} m', fontsize=10)
        ax.axis('off')
        provenance.append(dict(image_id=image_id, object_id=obj['object_id'], range_m=float(obj['range_m']),
                               original_sha256=sha(img_path), display_crop_xyxy=[x0, y0, x1, y1]))
    fig.text(.5, .03, 'Amber: supplied 2D box. Green: matched prediction. Enlarged crops; exploratory PC results.\n'
             'Source: DZSF / OSDaR23 (doi:10.57806/9mv146r0). Image crops and overlays: CC BY-SA 3.0 DE.',
             ha='center', fontsize=8)
    fig.subplots_adjust(bottom=.15, top=.85)
    fig.savefig(FIG / 'distance_examples.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    (REPORT / 'figure_provenance.json').write_text(json.dumps(dict(
        statistics_figures=['FigureS2_distance_coverage', 'FigureS3_monocular_validation'],
        qualitative_local_only=provenance,
        photo_licence='CC BY-SA 3.0 DE https://creativecommons.org/licenses/by-sa/3.0/de/',
        photo_notice='Display crops and overlays are derivatives under the same licence; not included in manuscript PDFs.'), indent=2) + '\n')


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    main()
