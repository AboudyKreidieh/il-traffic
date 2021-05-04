"""Tests for the features in core/alg.py."""
import unittest

from il_traffic import IntelligentDriverModel
from il_traffic import FollowerStopper
from il_traffic import PISaturation
from il_traffic import TimeHeadwayFollowerStopper
from il_traffic.core.experts import VelocityController


class TestIntelligentDriverModel(unittest.TestCase):
    """Tests for the IntelligentDriverModel object."""

    def test_init(self):
        """Validate the functionality of the __init__ method."""
        # Create the model.
        model = IntelligentDriverModel(
            v0=1,
            T=2,
            a=3,
            b=4,
            delta=5,
            s0=6,
            noise=7,
        )

        # Check the attributes.
        self.assertAlmostEqual(model.v0, 1)
        self.assertAlmostEqual(model.T, 2)
        self.assertAlmostEqual(model.a, 3)
        self.assertAlmostEqual(model.b, 4)
        self.assertAlmostEqual(model.delta, 5)
        self.assertAlmostEqual(model.s0, 6)
        self.assertAlmostEqual(model.noise, 7)

    def test_get_action(self):
        """Validate the functionality of the get_action method."""
        # Create the model.
        model = IntelligentDriverModel()

        # Test once.
        self.assertAlmostEqual(
            model.get_action(speed=5, headway=10, lead_speed=5),
            0.5092283950617285)

        # Test twice.
        self.assertAlmostEqual(
            model.get_action(speed=5, headway=10, lead_speed=0),
            -1.9613072882284588)


class TestVelocityController(unittest.TestCase):
    """Tests for the VelocityController object."""

    def test_init(self):
        """Validate the functionality of the __init__ method."""
        # Create the model.
        model = VelocityController(
            max_accel=1,
            max_decel=2,
            noise=3,
            sim_step=4,
        )

        # Check the attributes.
        self.assertAlmostEqual(model.max_accel, 1)
        self.assertAlmostEqual(model.max_decel, -2)
        self.assertAlmostEqual(model.noise, 3)
        self.assertAlmostEqual(model.sim_step, 4)

    def test_get_accel_from_v_des(self):
        """Validate the functionality of the get_accel_from_v_des method."""
        # Create the model.
        model = VelocityController(
            max_accel=1,
            max_decel=1,
            noise=0,
            sim_step=0.4,
        )

        # Test once.
        self.assertAlmostEqual(
            model._get_accel_from_v_des(speed=5, v_des=5.1), 0.25)

        # Test twice.
        self.assertAlmostEqual(
            model._get_accel_from_v_des(speed=5, v_des=4.9), -0.25)

        # Test three times.
        self.assertAlmostEqual(
            model._get_accel_from_v_des(speed=5, v_des=5.0), 0.0)


class TestFollowerStopper(unittest.TestCase):
    """Tests for the FollowerStopper object."""

    def test_init(self):
        """Validate the functionality of the __init__ method."""
        # Create the model.
        model = FollowerStopper(
            v_des=5.0,
            max_accel=1,
            max_decel=2,
            noise=3,
            sim_step=4,
        )

        # Check the attributes.
        self.assertAlmostEqual(model.v_des, 5.0)
        self.assertEqual(model.v_cmd, None)

    def test_get_action(self):
        """Validate the functionality of the get_action method."""
        # Create the model.
        model = FollowerStopper(
            v_des=5.,
            max_accel=1.,
            max_decel=1.,
            noise=0,
            sim_step=0.4,
        )

        # Test once.
        self.assertAlmostEqual(
            model.get_action(speed=4, headway=10, lead_speed=5),
            1.0)
        self.assertAlmostEqual(model.v_cmd, 5.0)

        # Test twice.
        self.assertAlmostEqual(
            model.get_action(speed=4, headway=10, lead_speed=0),
            -1.0)
        self.assertAlmostEqual(model.v_cmd, 0.0)


class TestPISaturation(unittest.TestCase):
    """Tests for the PISaturation object."""

    def test_init(self):
        """Validate the functionality of the __init__ method."""
        # Create the model.
        model = PISaturation(
            max_accel=1,
            max_decel=2,
            noise=3,
            sim_step=4,
            meta_period=5,
        )

        # Check the attributes.
        self.assertAlmostEqual(model.meta_period, 5)

    def test_get_action(self):
        """Validate the functionality of the get_action method."""
        # Create the model.
        model = PISaturation(
            max_accel=1.,
            max_decel=1.,
            noise=0,
            sim_step=0.4,
            meta_period=1,
        )

        # Test once.
        self.assertAlmostEqual(
            model.get_action(speed=4, headway=10, lead_speed=5),
            -1.0)

        # Test twice.
        self.assertAlmostEqual(
            model.get_action(speed=4, headway=10, lead_speed=0),
            -1.0)


class TestTimeHeadwayFollowerStopper(unittest.TestCase):
    """Tests for the TimeHeadwayFollowerStopper object."""

    def test_init(self):
        """Validate the functionality of the __init__ method."""
        # Create the model.
        model = TimeHeadwayFollowerStopper(
            v_des=5.0,
            max_accel=1,
            max_decel=2,
            noise=3,
            sim_step=4,
        )

        # Check the attributes.
        self.assertAlmostEqual(model.v_des, 5.0)
        self.assertEqual(model.v_cmd, None)

    def test_get_action(self):
        """Validate the functionality of the get_action method."""
        # Create the model.
        model = TimeHeadwayFollowerStopper(
            v_des=5.,
            max_accel=1.,
            max_decel=1.,
            noise=0,
            sim_step=0.4,
        )

        # Test once.
        self.assertAlmostEqual(
            model.get_action(speed=4, headway=10, lead_speed=5),
            1.0)
        self.assertAlmostEqual(model.v_cmd, 4.9367088607594924)

        # Test twice.
        self.assertAlmostEqual(
            model.get_action(speed=4, headway=10, lead_speed=0),
            -1.0)
        self.assertAlmostEqual(model.v_cmd, 0.0)


if __name__ == '__main__':
    unittest.main()
