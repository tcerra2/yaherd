# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import subprocess
import importlib.util

from boxmot.utils import REQUIREMENTS, logger


class TestRequirements():

    def check_requirements(self):
        requirements = [line.strip() for line in REQUIREMENTS.open() if line.strip() and not line.strip().startswith('#')]
        self.check_packages(requirements)

    def check_packages(self, requirements, cmds=''):
        """Test that each required package is available."""
        # Check packages and install missing ones

        s = ''  # missing packages
        for r in requirements:
            r = str(r)
            pkg_name = r.split()[0].split('>=')[0].split('==')[0].split('<')[0].split('>')[0].split('!')[0]
            try:
                importlib.util.find_spec(pkg_name)
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
