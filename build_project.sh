echo "Y" | pip uninstall gym_pybullet_drones
rm -rf dist/
poetry build
pip install dist/pybullet_multirotor-0.0.1-py3-none-any.whl