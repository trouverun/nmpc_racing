import argparse
import sys
from driver.driver import driver_process
from driver.rendering.render import render_process
from multiprocessing import Queue, Process, Event
from graph_tool import MainWindow
from PyQt6 import QtWidgets

parser = argparse.ArgumentParser()
parser.add_argument('--sim_path', type=str, default="/home/aleksi/Formula-Student-Driverless-Simulator/FSDS.sh")
parser.add_argument('--dynamics_type', type=str, default="kinematic_bicycle")
parser.add_argument('--mapping_from_scratch', type=bool, default=False)
parser.add_argument('--camera_disabled', type=bool, default=True)
parser.add_argument('--run_data_path', type=str, default='/home/aleksi/PycharmProjects/nmpc_racing/run_data/')
args = parser.parse_args()

render_queue = Queue()
graph_queue = Queue()
exit_event = Event()

map_list = ['TrainingMap', 'TrainingMap_reverse', 'CompetitionMapTestday2', 'CompetitionMap1']
driver = Process(
    target=driver_process,
    args=(
        args.sim_path, map_list, render_queue, graph_queue, exit_event, args.dynamics_type,
        args.mapping_from_scratch, args.camera_disabled, args.run_data_path
    )
)
driver.start()
renderer = Process(target=render_process, args=(render_queue, exit_event))
renderer.start()

app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
w.set_params(graph_queue)
w.resize(480, 1280)
w.show()
app.exec()
exit_event.set()
driver.join()
renderer.kill()
print("Killed")