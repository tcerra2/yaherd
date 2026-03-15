# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import subprocess

try:
    import pkg_resources
except ImportError:
    pkg_resources = None

from boxmot.utils import REQUIREMENTS, logger


class TestRequirements():

    def check_requirements(self):
        if pkg_resources is None:
            logger.warning("pkg_resources not available, skipping requirement checks")
            return
        requirements = pkg_resources.parse_requirements(REQUIREMENTS.open())
        self.check_packages(requirements)

    def check_packages(self, requirements, cmds=''):
        """Test that each required package is available."""
        # Ref: https://stackoverflow.com/a/45474387/
        
        if pkg_resources is None:
            logger.warning("pkg_resources not available, skipping package checks")
            return

        s = ''  # missing packages
        for r in requirements:
            r = str(r)
            try:
                pkg_resources.require(r)
            except Exception as e:
                logger.error(f'{e}')
                s += f'"{r}" '
        if s:
            logger.warning(f'\nMissing packages: {s}\nAtempting installation...')
            try:
                subprocess.check_output(f'pip install --no-cache {s} {cmds}', shell=True, stderr=subprocess.STDOUT)
            except Exception as e:
                logger.error(e)
                exit()
            logger.success('All the missing packages were installed successfully')
