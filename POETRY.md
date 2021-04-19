# Poetry init commands

---
## Install pyenv

```bash
curl https://pyenv.run | bash
```
add to your .bashrc oder .zshrc
```bash
export PATH="~/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

couple of needed libraries:
```bash
sudo apt update && sudo apt install -y make build-essential \
libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev \
libffi-dev liblzma-dev python-openssl git
```

Install specific python versions with pyenv
```bash
pyenv install 3.8.5
pyenv install 3.7.7
```
Init the python versions in your repo
```bash
pyenv local 3.8.5 3.7.7
```
## Install pipx
This is nice to have and installs libraries global
```bash
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```

## Install poetry

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
```

Add poetry to Path 
```bash
export PATH="$HOME/.poetry/bin:$PATH"
```

Init poetry
```bash
source ~/.poetry/env
poetry init --no-interaction
```

Build the package local
```bash
poetry install
```

Update poetry.lock
```bash
# update from changes inside poetry.lock
poetry update
# re install afterwards
poetry install
```

Build the package wheel
```bash
poetry build
```

Run a script from the pyproject.toml
```zsh
poetry run [script_name]
```

Open shell with functionality like virtual env. This is the same as `source ~/path_to_envs/env_name/bin/activate`
```zsh
poetry shell
```

## Nox functionality

```bash 
nox -rs lint
nox -rs black
```

```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```