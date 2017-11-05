# MI_webapp
Web application for the MI study

# How to run
1. Install [Conda](https://conda.io/docs/user-guide/install/index.html). Don't forget to add conda in your path. You can run commands below to add conda in your path.
  * `vim ~/.bashrc`
  * Add `export PATH=/home/YOUR_USER_NAME/anaconda3/bin:$PATH` in the end of the file.
  * Run `source ~/.bashrc`
2. Create Python Environment, and install required libraries:
  * Run `sh install.sh`
3. Run the following:
  * `source activate myenv`
  * `python manage.py runserver 0.0.0.0:8000`. If this port is occupied, try other ports.
