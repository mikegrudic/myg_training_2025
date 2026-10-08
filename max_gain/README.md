# Max-gain routes

The route finder is now the [vertmaxxer](https://github.com/mikegrudic/vertmaxxer) package (`~/code/vertmaxxer`);
see its README. Install it with `pip install -e ~/code/vertmaxxer`. `max_gain_route.py` and `max_gain/spurify.py`
here stand in for its `vertmaxxer` and `vertmaxxer-spurify` commands, using this repo's download cache.

This folder holds the survey built on it: `sweep_th.py` runs every trailhead, shape and distance, and `web/`
merges the runs and builds the route pages and the public site.
