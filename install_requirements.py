__author__ = 'matheusrosa'
import os
with open('requirements.txt', 'r') as f:
    reqs = f.read().split('\n')
    for req in reqs:
        if req.startswith('--') or req.startswith('#'):
            continue
        if len(req) > 0:
            os.system('easy_install %s' % req)