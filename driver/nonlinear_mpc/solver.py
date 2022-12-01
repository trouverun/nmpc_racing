from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from driver.nonlinear_mpc.dynamics import dynamics_model
import numpy as np
import config
from scipy import interpolate


class Solver:
    def __init__(self, dynamics_type):
        self.output_root = 'debug'
        self.N = config.mpc_horizon

        if dynamics_type not in ['kinematic_bicycle', 'dynamic_bicycle']:
            raise ValueError("unknown dynamics model type %s" % dynamics_type)

        self.dynamics_type = dynamics_type
        self.n_controls = 3    # u_steer, u_throttle, u_theta

        if dynamics_type == 'kinematic_bicycle':
            self.n_states = 8  # x, y, hdg, vel, slip, steer, throttle, theta
        else:
            self.n_states = 9  # x, y, hdg, vx, vy, w, steer, throttle, theta

        self.acados_solver = None
        self.f = None
        self.delay_compensation_f = None
        self._create_solver()
        self.initialized = False
        self.x0 = np.zeros([self.N+1, self.n_states])
        self.u0 = np.zeros([self.N, self.n_controls])
        self.last_initial_state = None
        self.last_midpoints = None

    def _create_solver(self):
        model, constraints, cost = dynamics_model(self.dynamics_type)
        self.f = model.f
        self.delay_compensation_f = model.delay_compensation_f
        ocp = AcadosOcp()
        ocp.dims.N = self.N

        model_ac = AcadosModel()
        model_ac.disc_dyn_expr = model.disc_dyn_expr
        model_ac.con_h_expr = constraints.expr
        model_ac.con_h_expr_e = constraints.expr
        model_ac.x = model.x
        model_ac.u = model.u
        model_ac.z = model.z
        model_ac.p = model.p
        model_ac.name = model.name
        ocp.model = model_ac

        # Costs:
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        ocp.model.cost_y_expr = cost.expr
        ocp.model.cost_y_expr_e = cost.expr_e
        ocp.cost.yref = cost.yref
        ocp.cost.yref_e = cost.yref_e
        ocp.cost.W = cost.W
        ocp.cost.W_e = cost.W_e

        # Control constraints
        ocp.constraints.lbu = constraints.lbu
        ocp.constraints.ubu = constraints.ubu
        ocp.constraints.idxbu = np.arange(model.u.size()[0])
        ocp.constraints.lsbu = constraints.lsbu
        ocp.constraints.usbu = constraints.usbu
        ocp.constraints.idxsbu = constraints.idxsbu

        # State constraints:
        ocp.constraints.lbx = constraints.lbx
        ocp.constraints.lbx_e = ocp.constraints.lbx
        ocp.constraints.ubx = constraints.ubx
        ocp.constraints.ubx_e = ocp.constraints.ubx
        ocp.constraints.idxbx = np.arange(model.x.size()[0])
        ocp.constraints.idxbx_e = ocp.constraints.idxbx

        # State soft constraints:
        ocp.constraints.lsbx = constraints.lsbx
        ocp.constraints.lsbx_e = constraints.lsbx_e
        ocp.constraints.usbx = constraints.usbx
        ocp.constraints.usbx_e = constraints.usbx_e
        ocp.constraints.idxsbx = constraints.idxsbx
        ocp.constraints.idxsbx_e = constraints.idxsbx_e

        # Nonlinear constraints
        ocp.constraints.lh = constraints.lh
        ocp.constraints.lh_e = ocp.constraints.lh
        ocp.constraints.uh = constraints.uh
        ocp.constraints.uh_e = ocp.constraints.uh

        # Nonlinear soft constraints
        ocp.constraints.lsh = constraints.lsh
        ocp.constraints.lsh_e = constraints.lsh_e
        ocp.constraints.ush = constraints.ush
        ocp.constraints.ush_e = constraints.ush_e
        ocp.constraints.idxsh = constraints.idxsh
        ocp.constraints.idxsh_e = constraints.idxsh_e

        # Constraint gradient and hessian diagonal (in order of: U, X, H)
        ocp.cost.zl = cost.zl
        ocp.cost.zl_e = cost.zl_e
        ocp.cost.zu = cost.zu
        ocp.cost.zu_e = cost.zu_e
        ocp.cost.Zl = cost.Zl
        ocp.cost.Zl_e = cost.Zl_e
        ocp.cost.Zu = cost.Zu
        ocp.cost.Zu_e = cost.Zu_e

        # Initial state constraint:
        ocp.constraints.x0 = np.zeros(model.x.size()[0])

        # Set params:
        ocp.parameter_values = np.zeros(model.p.size()[0])
        self.n_learned_params = model.n_learned_params

        ocp.solver_options.tf = config.mpc_fast_lap_dt * config.mpc_horizon
        # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        # ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        # ocp.solver_options.hpipm_mode = 'ROBUST'
        ocp.solver_options.nlp_solver_type = "SQP"
        # ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "DISCRETE"

        ocp.solver_options.qp_solver_ric_alg = 0
        ocp.solver_options.sim_method_jac_reuse = 0
        ocp.solver_options.sim_method_num_stages = 1
        ocp.solver_options.sim_method_num_steps = 1
        ocp.solver_options.nlp_solver_max_iter = config.solver_max_iter
        ocp.solver_options.print_level = 0
        ocp.solver_options.qp_tol = config.solver_tolerance
        # ocp.solver_options.levenberg_marquardt = 0.1
        self.acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

    def delay_compensation(self, state, dt):
        if self.dynamics_type == 'kinematic_bicycle':
            update = self.delay_compensation_f(state[2:], dt)
        else:
            update = self.delay_compensation_f(state[2:], 0, dt)
        update = np.asarray(update).flatten()
        updated_state = state
        updated_state[:-2] += update
        return updated_state

    def reset(self):
        self.initialized = False
        self.x0 = np.zeros([self.N+1, self.n_states])
        self.u0 = np.zeros([self.N, self.n_controls])
        self.last_initial_state = None
        self.last_midpoints = None

    def initialize(self, initial_state, midpoints, max_speed, dt):
        # Append path progress variable:
        initial_state = np.r_[initial_state.flatten(), 0]
        self.last_initial_state = initial_state
        self.acados_solver.set(0, "lbx", initial_state)
        self.acados_solver.set(0, "ubx", initial_state)

        k = 1
        if len(midpoints) > 2:
            k = config.spline_deg
        distances = config.b_spline_points
        cx_spline = interpolate.splrep(midpoints[:, 0], midpoints[:, 1], k=k)
        cy_spline = interpolate.splrep(midpoints[:, 0], midpoints[:, 2], k=k)
        c_x = interpolate.splev(distances, cx_spline)
        c_y = interpolate.splev(distances, cy_spline)
        c_dx = interpolate.splev(distances, cx_spline, der=1)
        c_dy = interpolate.splev(distances, cy_spline, der=1)
        self.last_midpoints = midpoints

        # Fill in the spline parameters, model approximation parameters and initial guesses
        for i in range(config.mpc_horizon + 1):
            p = np.r_[c_x, c_y, c_dx, c_dy, max_speed, dt]
            self.acados_solver.set(i, "p", p)
            if self.initialized:
                self.acados_solver.set(i, "x", self.x0[i])
                if i < config.mpc_horizon:
                    self.acados_solver.set(i, "u", self.u0[i])
            else:
                self.acados_solver.set(i, "x", initial_state)

    def solve(self):
        status = self.acados_solver.solve()
        steer, throttle = 0, -1
        if status in [0, 2]:
            self.initialized = True
            for i in range(config.mpc_horizon+1):
                x = self.acados_solver.get(i, "x")
                if i > 0:
                    self.x0[i - 1] = x
                    if i < config.mpc_horizon:
                        u = self.acados_solver.get(i, "u")
                        self.u0[i - 1] = u
                    if i == 1:
                        if self.dynamics_type == 'kinematic_bicycle':
                            steer = x[5]
                            throttle = x[6]
                        else:
                            steer = x[6]
                            throttle = x[7]
                        # We might terminate early with some constraint violations, clip to make sure:
                        steer = np.clip(steer, -1, 1)
                        throttle = np.clip(throttle, -1, 1)
            self.x0[-1] = self.x0[-2]
            return steer, throttle, self.x0[:-1]
        else:
            np.save('%s/initial_state' % self.output_root, self.last_initial_state)
            np.save('%s/midpoints' % self.output_root, self.last_midpoints)
            self.acados_solver.store_iterate('%s/corpse.json' % self.output_root)
            raise RuntimeError("Solver failed with status %d" % status)