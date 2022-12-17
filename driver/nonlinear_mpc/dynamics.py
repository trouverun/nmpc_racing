from casadi import *
import config
import scipy

fake_inf = 1E7


def dynamics_model(dynamics_type):
    cost = types.SimpleNamespace()
    constraints = types.SimpleNamespace()
    model = types.SimpleNamespace()

    # Controls
    u_steer = MX.sym("usteer")
    u_throttle = MX.sym("uthrottle")
    u_theta = MX.sym("utheta")
    u = vertcat(u_steer, u_throttle, u_theta)

    # Helpers
    dt = MX.sym("dt", 1)
    max_speed = MX.sym("max_speed", 1)
    model.n_learned_params = 0

    # Dynamics definitions
    if dynamics_type == 'kinematic_bicycle':
        # State
        x = MX.sym("x")
        y = MX.sym("y")
        hdg = MX.sym("hdg")
        v = MX.sym("v")
        slip = MX.sym("slip")
        steer = MX.sym("steer")
        throttle = MX.sym("throttle")
        theta = MX.sym("theta")
        state = vertcat(x, y, hdg, v, slip, steer, throttle, theta)
        # Define kinematic bicycle model
        input_state = vertcat(hdg, v, slip, steer, throttle)
        f_expr = vertcat(
            v * cos(hdg + slip),  # x
            v * sin(hdg + slip),  # y
            v / config.car_lr * sin(slip),  # psi
            throttle * config.car_max_acceleration - config.car_rolling_resistance - sign(v)*config.car_drag*v**2,  # v
            # config.car_lr * steer*config.car_max_steer / (config.car_lf + config.car_lr),  # slip
            atan2(config.car_lr * tan(steer*config.car_max_steer), (config.car_lf + config.car_lr)),  # slip
            # concat output with zeroes, since steer and throttle are part of input but not part of the output:
            0,
            0
        )
        f = Function('f', [input_state], [f_expr])
        # Discretize with RK4:
        k1 = f(input_state)
        # crop the x and y since they are part of the output, but not part of the input
        k2 = f(input_state + dt / 2 * k1[2:])
        k3 = f(input_state + dt / 2 * k2[2:])
        k4 = f(input_state + dt * k3[2:])
        dynamics = dt/6*(k1+2*k2+2*k3+k4)
        dynamics = dynamics[:-2]  # remove steer and throttle since they are dealt by simple euler dynamics below
        model.delay_compensation_f = Function("kinematic_delay_comp", [input_state, dt], [dynamics])
    else:
        # Dynamic bicycle model and neural dynamic share the same state:
        x = MX.sym("x")
        y = MX.sym("y")
        hdg = MX.sym("hdg")
        w = MX.sym("w")
        vx = MX.sym("vx")
        vy = MX.sym("vy")
        steer = MX.sym("steer")
        throttle = MX.sym("throttle")
        theta = MX.sym("theta")
        state = vertcat(x, y, hdg, vx, vy, w, steer, throttle, theta)
        if dynamics_type == 'dynamic_bicycle':
            input_state = vertcat(hdg, vx, vy, w, steer, throttle)
            ar = atan2(vy - w*config.car_lr, vx)
            Fry = config.wheel_Dr * sin(config.wheel_Cr * arctan(config.wheel_Br * ar))
            af = atan2(vy + w*config.car_lf, vx) - steer*config.car_max_steer
            Ffy = config.wheel_Df * sin(config.wheel_Cf * arctan(config.wheel_Bf * af))
            Frx = config.car_Tm * throttle - config.car_Tr0 - config.car_Tr2 * vx**2
            f_d_expr = vertcat(
                vx * cos(hdg) + vy*sin(hdg),
                vx * sin(hdg) + vy*cos(hdg),
                w,
                1/config.car_mass * Frx,
                1/config.car_mass * (Ffy*cos(steer*config.car_max_steer) + Fry),
                1/config.car_inertia * (Ffy*config.car_lf*cos(steer*config.car_max_steer) - Fry*config.car_lr),
                0,
                0
            )
            f_k_expr = vertcat(
                vx * cos(hdg) + vy * sin(hdg),
                vx * sin(hdg) + vy * cos(hdg),
                w,
                Frx / config.car_mass,
                (u_steer*config.car_max_steer*vx + steer*config.car_max_steer*(Frx/config.car_mass)) * (config.car_lr/(config.car_lr + config.car_lf)),
                (u_steer*config.car_max_steer*vx + steer*config.car_max_steer*(Frx/config.car_mass)) * (1 / (config.car_lr + config.car_lf)),
                0,
                0
            )
            vb_min = config.blend_min_speed
            vb_max = config.blend_max_speed
            lam = fmin(fmax((vx - vb_min) / (vb_max - vb_min), 0), 1)
            f_expr = lam*f_d_expr + (1-lam)*f_k_expr
            f = Function('f', [input_state, u_steer], [f_expr])
            # Discretize with RK4:
            k1 = f(input_state, u_steer)
            # crop the x, y and hdg since they are part of the output, but not part of the input
            k2 = f(input_state + dt / 2 * k1[2:], u_steer)
            k3 = f(input_state + dt / 2 * k2[2:], u_steer)
            k4 = f(input_state + dt * k3[2:], u_steer)
            dynamics = dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
            dynamics = dynamics[:-2]  # remove steer and throttle since they are dealt by simple euler dynamics below
            model.delay_compensation_f = Function("dynamic_delay_comp", [input_state, u_steer, dt], [dynamics])

    # Shared dynamics, control states are discretized with simple euler:
    f_d = state + vertcat(
        dynamics,
        dt * u_steer,
        dt * u_throttle,
        dt * u_theta,
    )

    # Control bounds:
    u_steer_min = -config.u_steer_max
    u_steer_max = config.u_steer_max
    u_throttle_min = -config.u_throttle_max
    u_throttle_max = config.u_throttle_max
    u_theta_min = 0
    u_theta_max = config.u_theta_max
    constraints.lbu = np.array([u_steer_min, u_throttle_min, u_theta_min])
    constraints.ubu = np.array([u_steer_max, u_throttle_max, u_theta_max])
    constraints.idxbu = np.arange(u.size()[0])
    constraints.lsbu = np.array([0])
    constraints.usbu = np.array([0])
    constraints.idxsbu = np.array([2])

    # Parametric splines for the track center line:
    theta0 = config.b_spline_points
    # center x position
    theta_cx = interpolant("theta_cx", "linear", [theta0])
    cx0 = MX.sym("cx0", config.n_bspline_points)
    cx_interp_exp = theta_cx(theta, cx0)
    cx_fun = Function('cx_fun', [theta, cx0], [cx_interp_exp])
    # center y position
    theta_cy = interpolant("theta_cy", "linear", [theta0])
    cy0 = MX.sym("cy0", config.n_bspline_points)
    cy_interp_exp = theta_cy(theta, cy0)
    cy_fun = Function('cy_fun', [theta, cy0], [cy_interp_exp])
    # center dx:
    theta_cdx = interpolant("theta_cdx", "linear", [theta0])
    cdx0 = MX.sym("cdx0", config.n_bspline_points)
    cdx_interp_exp = theta_cdx(theta, cdx0)
    cdx_fun = Function('cxd_fun', [theta, cdx0], [cdx_interp_exp])
    # center dy:
    theta_cdy = interpolant("theta_cdy", "linear", [theta0])
    cdy0 = MX.sym("cdy0", config.n_bspline_points)
    cdy_interp_exp = theta_cdy(theta, cdy0)
    cdy_fun = Function('cdy_fun', [theta, cdy0], [cdy_interp_exp])

    if dynamics_type == 'kinematic_bicycle':
        # State constraints:
        x_min = -fake_inf
        x_max = fake_inf
        y_min = -fake_inf
        y_max = fake_inf
        hdg_min = -fake_inf
        hdg_max = fake_inf
        v_min = 0
        v_max = config.car_max_speed
        slip_min = -config.car_max_slip
        slip_max = config.car_max_slip

        lbx = np.array([x_min, y_min, hdg_min, v_min, slip_min])
        ubx = np.array([x_max, y_max, hdg_max, v_max, slip_max])

        lsbx = np.array([0, 0, 0])
        lsbx_e = lsbx
        usbx = np.array([0, 0, 0])
        usbx_e = usbx
        idxsbx = np.array([3, 4, 7])
        idxsbx_e = idxsbx

        # Nonlinear constraints
        lh = np.array([0])
        uh = np.array([fake_inf])

        speed_violation = max_speed - v
        expr = vertcat(speed_violation)

        lsh = np.array([0, 0])
        lsh_e = lsh
        ush = np.array([0, 0])
        ush_e = ush
        idxsh = np.array([0, 1])
        idxsh_e = idxsh

        zl = 1 * np.array([1, 1, 1, 1, 1, 1])
        zl_e = zl[1:]
        zu = 1 * np.array([1, 1, 1, 1, 1, 1])
        zu_e = zu[1:]
        Zl = 1 * np.array([
            config.soft_u_theta_weight, config.soft_state_v_weight, config.soft_state_slip_weight,
            config.soft_state_theta_weight, config.soft_nl_max_v_weight, config.soft_nl_track_circle_weight
        ])
        Zl_e = Zl[1:]
        Zu = 1 * np.array([
            config.soft_u_theta_weight, config.soft_state_v_weight, config.soft_state_slip_weight,
            config.soft_state_theta_weight, config.soft_nl_max_v_weight, config.soft_nl_track_circle_weight
        ])
        Zu_e = Zu[1:]
    else:
        x_min = -fake_inf
        x_max = fake_inf
        y_min = -fake_inf
        y_max = fake_inf
        hdg_min = -fake_inf
        hdg_max = fake_inf
        w_min = -fake_inf
        w_max = fake_inf
        vx_min = 0
        vx_max = fake_inf
        vy_min = -fake_inf
        vy_max = fake_inf

        lbx = np.array([x_min, y_min, hdg_min, vx_min, vy_min, w_min])
        ubx = np.array([x_max, y_max, hdg_max, vx_max, vy_max, w_max])

        lsbx = np.array([0, 0])
        lsbx_e = lsbx
        usbx = np.array([0, 0])
        usbx_e = usbx
        idxsbx = np.array([3, 8])
        idxsbx_e = idxsbx

        # Nonlinear constraints
        lh = np.array([0])
        uh = np.array([fake_inf])
        speed_violation = max_speed - vx
        expr = vertcat(speed_violation)

        lsh = np.array([0, 0])
        lsh_e = lsh
        ush = np.array([0, 0])
        ush_e = ush
        idxsh = np.array([0, 1])
        idxsh_e = idxsh

        zl = 10 * np.array([1, 1, 1, 1, 1])
        zl_e = zl[1:]
        zu = 10 * np.array([1, 1, 1, 1, 1])
        zu_e = zu[1:]
        Zl = 1 * np.array([
            config.soft_u_theta_weight, config.soft_state_v_weight, config.soft_state_theta_weight,
            config.soft_nl_max_v_weight, config.soft_nl_track_circle_weight
        ])
        Zl_e = Zl[1:]
        Zu = 1 * np.array([
            config.soft_u_theta_weight, config.soft_state_v_weight, config.soft_state_theta_weight,
            config.soft_nl_max_v_weight, config.soft_nl_track_circle_weight
        ])
        Zu_e = Zu[1:]

    p = vertcat(cx0, cy0, cdx0, cdy0, max_speed, dt)

    # Shared state constraints
    steer_min = -1
    steer_max = 1
    throttle_min = -1
    throttle_max = 1
    theta_min = 0
    theta_max = config.bspline_max_distance
    constraints.lbx = np.r_[lbx, np.array([steer_min, throttle_min, theta_min])]
    constraints.ubx = np.r_[ubx, np.array([steer_max, throttle_max, theta_max])]
    constraints.lsbx = lsbx
    constraints.lsbx_e = lsbx_e
    constraints.usbx = usbx
    constraints.usbx_e = usbx_e
    constraints.idxsbx = idxsbx
    constraints.idxsbx_e = idxsbx_e

    # Shared nonlinear constraints:
    constraints.lh = np.r_[lh, np.array([-fake_inf])]
    constraints.uh = np.r_[uh, np.array([config.track_radius ** 2])]
    center_circle_deviation = (x - cx_fun(theta, cx0)) ** 2 + (y - cy_fun(theta, cy0)) ** 2
    constraints.expr = vertcat(expr, center_circle_deviation)
    constraints.lsh = lsh
    constraints.lsh_e = lsh_e
    constraints.ush = ush
    constraints.ush_e = ush_e
    constraints.idxsh = idxsh
    constraints.idxsh_e = idxsh_e

    # Costs
    cost.zl = zl
    cost.zl_e = zl_e
    cost.zu = zu
    cost.zu_e = zu_e
    cost.Zl = Zl
    cost.Zl_e = Zl_e
    cost.Zu = Zu
    cost.Zu_e = Zu_e
    phi = atan2(cdy_fun(theta, cdy0), cdx_fun(theta, cdx0))
    e_contour = sin(phi) * (x - cx_fun(theta, cx0)) - cos(phi) * (y - cy_fun(theta, cy0))
    e_lag = -cos(phi) * (x - cx_fun(theta, cx0)) - sin(phi) * (y - cy_fun(theta, cy0))
    cost.expr = vertcat(e_contour, e_lag, theta, u_steer, u_throttle, steer, throttle)
    cost.expr_e = vertcat(e_contour, e_lag)
    cost.yref = np.array([0, 0, 0, 0, 0, 0, 0])
    cost.yref_e = np.array([0, 0])
    cost.W = scipy.linalg.block_diag(
        config.contour_weight, config.lag_weight, -config.theta_weight,
        config.u_steer_weight, config.u_throttle_weight, config.steer_weight, config.throttle_weight
    )
    cost.W_e = scipy.linalg.block_diag(config.contour_weight, config.lag_weight)

    params = types.SimpleNamespace()
    model.disc_dyn_expr = f_d
    model.f = f
    model.x = state
    model.u = u
    model.z = vertcat([])
    model.p = p
    model.name = "car_%s_dynamics" % dynamics_type
    model.params = params

    return model, constraints, cost