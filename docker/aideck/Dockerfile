FROM bitcraze/aideck:latest

# gapy's image generation needs numpy 1.22.x. Upstream's make-example pip-installs
# this on every build; bake it into the image instead so our in-place builds are
# offline and don't reinstall it each run.
RUN pip3 install --no-cache-dir numpy==1.22.3

WORKDIR /module
