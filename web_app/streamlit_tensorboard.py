from pathlib import Path
from tensorboard import manager
import streamlit.components.v1 as components  # type: ignore
import shlex
import random
import html
import json
import subprocess
import os


def st_tensorboard(logdir="/logs/", port=8530, width=None, height=800, scrolling=True):
    """Embed Tensorboard within a Streamlit app
    Parameters
    ----------
    logdir: str
        Directory where TensorBoard will look to find TensorFlow event files that it can display.
        TensorBoard will recursively walk the directory structure rooted at logdir, looking for .*tfevents.* files.
        Defaults to /logs/
    port: int
        Port to serve TensorBoard on. Defaults to 6006
    width: int
        The width of the frame in CSS pixels. Defaults to the report’s default element width.
    height: int
        The height of the frame in CSS pixels. Defaults to 800.
    scrolling: bool
        If True, show a scrollbar when the content is larger than the iframe.
        Otherwise, do not show a scrollbar. Defaults to True.
    Example
    -------
    >>> st_tensorboard(logdir="/logs/", port=6006, width=1080)
    """

    subprocess.run("kill $(ps -e | grep 'tensorboard' | awk '{print $1}')", shell=True)
    # replace with the port number you want to shut down

    # Find the PID of the process running on port 6006
    pid_command = "lsof -i:8530 | awk '{print $2}' | sed -n 2p"
    pid = os.popen(pid_command).read().strip()

    # Kill the process with the given PID
    kill_command = f"kill -9 {pid}"
    os.system(kill_command)

    logdir = Path(str(logdir)).as_posix()
    port = port
    width = width
    height = height

    frame_id = "tensorboard-frame-{:08x}".format(random.getrandbits(64))
    shell = """
        <iframe id="%HTML_ID%" width="100%" height="%HEIGHT%" frameborder="0">
        </iframe>
        <script>
        (function() {
            const frame = document.getElementById(%JSON_ID%);
            const url = new URL(%URL%, window.location);
            const port = %PORT%;
            if (port) {
            url.port = port;
            }
            frame.src = url;
        })();
        </script>
    """

    args_string = f"--logdir {logdir} --port {port}"
    parsed_args = shlex.split(args_string, comments=True, posix=True)
    start_result = manager.start(parsed_args)

    if isinstance(start_result, manager.StartReused):
        port = start_result.info.port
        print(f"Reusing TensorBoard on port {port}")

    proxy_url = "http://localhost:%PORT%"

    proxy_url = proxy_url.replace("%PORT%", "%d" % port)
    replacements = [
        ("%HTML_ID%", html.escape(frame_id, quote=True)),
        ("%JSON_ID%", json.dumps(frame_id)),
        ("%HEIGHT%", "%d" % height),
        ("%PORT%", "0"),
        ("%URL%", json.dumps(proxy_url)),
    ]

    for (k, v) in replacements:
        shell = shell.replace(k, v)

    return components.html(shell, width=width, height=height, scrolling=scrolling)
