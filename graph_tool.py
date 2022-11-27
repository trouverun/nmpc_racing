from PyQt6 import QtWidgets, QtCore
import numpy as np
import pyqtgraph as pg


class MainWindow(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.timing_queue = None
        self.v_layout = QtWidgets.QVBoxLayout()
        self.h_layout = QtWidgets.QHBoxLayout()

        # Build the plots:
        self.graph_layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.Direction.LeftToRight)
        self.graph = pg.GraphicsLayoutWidget()
        self.graph_layout.addWidget(self.graph)
        self.n_plots = 13
        self.plot_history_len = 100
        self.x = np.repeat(np.arange(self.plot_history_len), self.n_plots).reshape(self.plot_history_len, self.n_plots).T
        self.y = np.zeros([self.n_plots, self.plot_history_len])
        self.plots = []
        self.plot_items = []
        plot_i = 0
        item_i = 0
        for title, lb, ub, items in [
            ("linear vel", -15, 15, [
               ((255, 0, 0), "vx"),  ((0, 255, 0), "vy"),
            ]),
            ("linear acc", -20, 20, [
                ((255, 0, 0), "ax"), ((0, 255, 0), "ay"),
            ]),
            ("angular", -60, 60, [
                ((255, 0, 0), "w"),  ((0, 255, 0), "gt_w"),
            ]),
            ("control", -1, 1, [
                ((255, 0, 0), "steer"), ((0, 255, 0), "throttle")
            ]),
            ("sim", 0, 60, [
                ((255, 0, 0), "frame_time")
            ]),
            ("vision time", 0, 75, [
                ((255, 0, 0), "vision_time")
            ]),
            ("mpc time", 0, 75, [
                ((255, 0, 0), "solve_time")
            ]),
            ("total time", 0, 125, [
                ((255, 0, 0), "iteration_time"), ((0, 255, 0), "solver_dt")
            ]),
        ]:
            self.plots.append(self.graph.addPlot(row=plot_i, col=0))
            self.plots[plot_i].setYRange(lb, ub)
            self.plots[plot_i].setTitle(title)
            self.plots[plot_i].showGrid(True, True)
            self.plots[plot_i].addLegend()
            for color, name in items:
                self.plot_items.append(
                    self.plots[plot_i].plot(
                        self.x[item_i], self.y[item_i], pen=pg.mkPen(color=color), name=name
                    )
                )
                item_i += 1
            plot_i += 1

        self.v_layout.addLayout(self.graph_layout)
        self.h_layout.addLayout(self.v_layout)
        self.setLayout(self.h_layout)

        self.timer = QtCore.QTimer()
        self.timer.setInterval(5)
        self.timer.timeout.connect(self.update_data)
        self.timer.start()

    def set_params(self, timing_queue):
        self.timing_queue = timing_queue

    def update_data(self):
        try:
            inputs = self.timing_queue.get(block=False)
        except Exception:
            return

        i = 0
        for values_list in inputs.values():
            for value in values_list:
                self.y[i, 0:-1] = self.y[i, 1:]
                self.y[i, -1] = value
                self.plot_items[i].setData(self.x[i], self.y[i])
                i += 1