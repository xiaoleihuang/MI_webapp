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
4. Open browswer and open `0.0.0.0:8000` for homepage.
5. `0.0.0.0:8000/topics/` for topic language analysis

# Screenshots of demo
1. Only Content
![Content Evaluation](./resources/screenshots/1.png?raw=true "Optional Title" | width=80%)
2. With context
![Contextual Mode](./resources/screenshots/2.png?raw=true "Optional Title")

# License
1. This project is licensed under the GPL License - see the [LICENSE.md](https://github.com/xiaoleihuang/MI_webapp/blob/master/LICENSE) file for details

# Contact
This author.
