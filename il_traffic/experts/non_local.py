"""A controller that attempts to drive at average downstream speeds."""


class NonLocalHarmonizer(ExpertModel):
    """A controller that attempts to drive at average downstream speeds.

    Attributes
    ----------
    v_target : float
        target (estimated) desired speed
    v_max : float
        maximum assigned speed
    sim_step : float
        simulation time step, in sec/step
    c1 : float
        scaling term for the response to the proportional error
    c2 : float
        scaling term for the response to the differential error
    th_target : float
        target time headway
    sigma : float
        standard deviation for the Gaussian smoothing kernel
    """

    def __init__(self, sim_step):
        """Instantiate the controller.

        Parameters
        ----------
        sim_step : float
            simulation time step, in sec/step
        """
        super(NonLocalHarmonizer, self).__init__(noise=0.)

        # simulation time step

        # Follower-Stopper wrapper
        self.fs = TimeHeadwayFollowerStopper(v_des=30., sim_step=sim_step)

        # controller parameters
        self.v_target = 30.
        self.v_max = 40.
        self.sim_step = sim_step
        self._prev_th = None
        self._vl = deque(maxlen=10)

        # tunable parameters
        self.c1 = 1.0
        self.c2 = 0.0
        self.th_target = 2.
        self.sigma = 3000.

    @staticmethod
    def kernelsmooth(x0, x, z, sigma):
        """Return a kernel-smoothing average of downstream speeds."""
        densities = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(
            -np.square(x - x0) / (2 * sigma ** 2))

        densities = densities / sum(densities)

        return sum(densities * z)

    def get_action(self, speed, headway, lead_speed, **kwargs):
        """See parent class."""
        avg_speed = kwargs.get("avg_speed", None)
        segments = kwargs.get("segments", None)
        x0 = kwargs.get("pos", None)

        # Update the buffer of lead speeds.
        self._vl.append(lead_speed)

        if avg_speed is not None:
            # Collect relevant traffic-state info.
            # - ix0: segments in front of the ego vehicle
            # - ix1: final segment, or clipped to estimate congested speeds
            ix0 = max(next(i for i in range(len(segments)) if segments[i]>=x0) - 2, 0)
            try:
                ix1 = next(j for j in range(ix0 + 2, len(segments)) if avg_speed[j] - np.mean(avg_speed[ix0:j]) > 15) + 1
            except StopIteration:
                ix1 = len(segments)
            segments = segments[ix0:ix1]
            avg_speed = avg_speed[ix0:ix1]

            th = headway / (speed + 1e-6)
            th = max(0., min(40., th))
            th_error = th - self.th_target

            if self._prev_th is None:
                delta_th_error = 0.
            else:
                delta_th_error = (self._prev_th - th) / self.sim_step
                self._prev_th = th

            self.v_target = \
                self.kernelsmooth(x0, segments, avg_speed, self.sigma) + \
                self.c1 * th_error + \
                self.c2 * delta_th_error

        # Update desired speed.
        max_decel = -1.0
        max_accel = 1.0
        prev_vdes = self.fs.v_des
        self.fs.v_des = max(
            prev_vdes + max_decel * self.sim_step,
            min(
                prev_vdes + max_accel * self.sim_step,
                self.v_target,
            )
        )

        # Keep within reasonable bounds.
        self.fs.v_des = max(0., min(40., self.fs.v_des))

        # Return acceleration command by follower stopper.
        return self.fs.get_action(
            speed=speed,
            lead_speed=lead_speed,
            headway=headway,
        )
