#! /usr/bin/env python

import argparse, logging
import sys
import numpy as np
import zmq

import rr100_rl
from sbx import CrossQ

class RLController:
    '''
    Actual RLController class.
    Contains an ZMQ REP server to respond to the ROS RLControllerBridge node.
    This was done because of version conflicts between ROS noetic (python 3.8.x)
    and the minimum required version for JAX-based RL frameworks (min. python 3.10.x)
    '''

    def __init__(
        self,
        model,
    ) -> None:
        logging.info("Starting RL controller...")
        self.model = model

        logging.info("Creating ZMQ context and socket...")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        # self.socket.setsockopt(zmq.LINGER, 0)
        # self.socket.setsockopt(zmq.IMMEDIATE, 1)

        # self.server_uri = f"tcp://127.0.0.1:{rr100_rl.CONTROLLER_PORT}"
        self.server_uri = f"ipc:///tmp/rl_controller.pipe"
        self.socket.bind(self.server_uri)
        logging.info(f"Socket bound to {self.server_uri}")

    def end(self):
      self.socket.unbind(self.server_uri)
      self.socket.close()
      self.context.term()

    def spin(self):
        while True:
            logging.info("Waiting for client request...")
            params : dict = self.socket.recv_json() # type: ignore
            logging.debug(f"Received request : {params}")
            action, lstm_state = self.model.predict(
                np.array(params["observation"]),
                episode_start=params["episode_start"],
                deterministic=params["deterministic"]
            )
            response = {
                "action" : action.tolist(),
                "lstm_state" : lstm_state,
            }
            logging.debug(f"Responding to request with {response}")
            self.socket.send_json(response)
            logging.info("Response, sent")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-path", "-m", type=str, required=True)
  parser.add_argument("--device", "-d", type=str, choices=["cuda", "cpu", "auto"], default="cuda")
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO) # default logging config for now

  model = CrossQ.load(path=args.model_path, device="cuda", buffer_size=1, seed=None)
  controller = RLController(model)
  try:
    controller.spin()
  except KeyboardInterrupt:
    print("Exiting...")
  finally:
    controller.end()
