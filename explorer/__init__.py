"""Interactive explorer for the Project 07 transmission chains.

`python scripts/explore.py` starts a local server and opens a browser. The page shows the transmitter, the
channel and the receiver as a map, and lets the user step through every stage of a real transmission: the
stages are computed by the project's own code (`semcom.*`), not re-implemented or faked.

Modules
    catalog  the systems, images and options offered by the UI
    trace    runs one transmission and records every stage (the physics)
    encode   turns tensors into things a browser can draw (PNG, latent maps, symbol clouds)
    server   stdlib HTTP server and JSON API
"""
