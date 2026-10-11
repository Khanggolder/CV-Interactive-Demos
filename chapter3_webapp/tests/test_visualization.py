import unittest

from utils.ransac import generate_dataset, guided_dataset, run_ransac
from utils.visualization import LAB_STAGES, lab_stage_figure, point_cloud_figure


class VisualizationTests(unittest.TestCase):
    def test_extreme_generated_points_remain_inside_chart_axes(self) -> None:
        points = generate_dataset(100, 100, 0.5, seed=0).points
        figure = point_cloud_figure(points, title="Extreme parameters")
        self.assertLessEqual(figure.layout.xaxis.range[0], points[:, 0].min())
        self.assertGreaterEqual(figure.layout.xaxis.range[1], points[:, 0].max())
        self.assertLessEqual(figure.layout.yaxis.range[0], points[:, 1].min())
        self.assertGreaterEqual(figure.layout.yaxis.range[1], points[:, 1].max())

    def test_lab_stages_reveal_only_computed_geometry(self) -> None:
        points = guided_dataset().points
        result = run_ransac(points, epsilon=0.22, seed=7647, iterations=3)
        figures = {
            stage: lab_stage_figure(points, result, 0.22, stage)
            for stage in LAB_STAGES
        }
        names = {stage: [trace.name for trace in figure.data] for stage, figure in figures.items()}
        self.assertNotIn("Candidate hiện tại", names["Sample"])
        self.assertIn("Candidate hiện tại", names["Fit"])
        self.assertNotIn("Final TLS refit", names["Fit"])
        self.assertIn("Dải ε = 0.22", names["Measure"])
        self.assertLessEqual(names["Measure"].count("Khoảng cách vuông góc"), 2)
        self.assertIn("Candidate hiện tại", names["Consensus"])
        self.assertIn("Best consensus model", names["Update Best"])
        self.assertIn("Final TLS refit", names["Update Best"])
        marker_trace = next(trace for trace in figures["Consensus"].data if trace.name == "Điểm dữ liệu")
        self.assertEqual(len(marker_trace.marker.color), len(points))
        self.assertEqual(sum(color == "#1996c8" for color in marker_trace.marker.color), result.trials[-1].score)

    def test_presentation_figure_has_larger_labels_and_markers(self) -> None:
        points = guided_dataset().points
        normal = point_cloud_figure(points, title="Normal")
        projected = point_cloud_figure(points, title="Projected", presentation=True)
        self.assertGreater(projected.layout.height, normal.layout.height)
        self.assertGreater(projected.layout.font.size, normal.layout.font.size)
        self.assertGreater(projected.data[0].marker.size[0], normal.data[0].marker.size[0])


if __name__ == "__main__":
    unittest.main()
