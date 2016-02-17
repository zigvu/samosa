#### Install dependencies

> for req in $(cat requirements.txt); do sudo pip install $req; done

#### Delete all pyc files

> find . -name '*.pyc' -delete
