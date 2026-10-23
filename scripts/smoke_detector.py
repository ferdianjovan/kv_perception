#!/usr/bin/env python3
"""ROS executable for the smoke detector."""

import faulthandler

from kv_perception.smoke_detector import main


if __name__ == '__main__':
    faulthandler.enable(all_threads=True)
    main()
