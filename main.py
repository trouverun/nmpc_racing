import argparse
import sys
from driver.driver import driver_process
from driver.rendering.render import render_process
from multiprocessing import Queue, Process, Event
from graph_tool import MainWindow
from PyQt6 import QtWidgets

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sim_path', type=str, default="/home/aleksi/Formula-Student-Driverless-Simulator/FSDS.sh")
args = parser.parse_args()

render_queue = Queue()
graph_queue = Queue()
exit_event = Event()

# main_model = 'kinematic_bicycle'
main_model ='dynamic_bicycle'

map_list = ['TrainingMap', 'TrainingMap_reverse', 'CompetitionMapTestday2', 'CompetitionMap1']
driver = Process(target=driver_process, args=(args.sim_path, map_list, render_queue, graph_queue, exit_event, main_model))
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