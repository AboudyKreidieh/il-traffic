"""Tests for the features in utils/."""
import unittest
import os
import shutil
import pandas as pd
import csv

import il_traffic.config as config
from il_traffic.utils.visualize import process_emission
from il_traffic.utils.misc import dict_update


class TestMisc(unittest.TestCase):
    """Tests for the features in core/utils/misc.py."""

    def test_dict_update(self):
        """Validate the functionality of the dict_update method."""
        dict1 = {"hello": {"world": {"1": "foo"}}}
        dict2 = {"hello": {"world": {"2": "bar"}}}

        self.assertDictEqual(
            dict_update(dict1, dict2),
            {"hello": {"world": {"1": "foo", "2": "bar"}}}
        )


class TestVisualize(unittest.TestCase):
    """Tests for the features in core/utils/visualize.py."""

    def test_process_emission(self):
        """Validate the functionality of the process_emission method."""
        directory = os.path.join(
            config.PROJECT_PATH, "tests/test_files/process_emission")
        fp = os.path.join(directory, "emission.csv")
        if not os.path.exists(directory):
            os.mkdir(directory)

        # Create an emission file with more degrees of precision than needed.
        d = {
            'time': [0.123456789],
            'id': [0.987654321],
            'type': [7],
            'speed': [7],
            'headway': [7],
            'target_accel_with_noise_with_failsafe': [7],
            'target_accel_no_noise_no_failsafe': [7],
            'target_accel_with_noise_no_failsafe': [7],
            'target_accel_no_noise_with_failsafe': [7],
            'realized_accel': [7],
            'edge_id': [7],
            'lane_number': [7],
            'relative_position': [7],
        }
        with open(fp, "w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(d.keys())
            writer.writerows(zip(*d.values()))

        # Run the process_emission method and read the precision of the new
        # emission file.
        process_emission(directory)
        df = pd.read_csv(fp)
        self.assertAlmostEqual(df.time[0], 0.123)
        self.assertAlmostEqual(df.id[0], 0.988)
        self.assertAlmostEqual(df.speed[0], 7)

        # Delete the created file.
        shutil.rmtree(directory)


if __name__ == '__main__':
    unittest.main()
