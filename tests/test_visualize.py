import numpy as np
import SimpleITK as sitk

from branchseed.models import DaughterPrediction
from branchseed.visualize import save_orthogonal_overlays


def test_visualizer_saves_headless_orthogonal_figure(tmp_path):
    image = sitk.GetImageFromArray(np.arange(8 * 9 * 10).reshape(8, 9, 10))
    image.SetSpacing((0.5, 0.75, 1.25))
    overlay = sitk.GetImageFromArray(np.zeros((8, 9, 10), dtype=np.uint8))
    overlay[3:7, 3:6, 2:5] = 1
    overlay.CopyInformation(image)
    prediction = DaughterPrediction(
        ostium_xyz=image.TransformIndexToPhysicalPoint((4, 4, 4)),
        direction_xyz=(2, 0, 0),
        candidate_path_xyz=(
            image.TransformIndexToPhysicalPoint((2, 2, 2)),
            image.TransformIndexToPhysicalPoint((5, 5, 5)),
        ),
    )
    destination = tmp_path / "orthogonal.png"

    returned = save_orthogonal_overlays(
        image,
        destination,
        overlay=overlay,
        predictions=[prediction],
    )

    assert returned == destination
    assert destination.is_file()
    assert destination.stat().st_size > 0
