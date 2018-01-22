# MI_webapp
Web application for the MI study

# Tested Platform
1. OS: Ubuntu
2. Browser: Chrome, Firefox
3. Python: 2.7+

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
4. Open browser and open `0.0.0.0:8000` for homepage.
5. `0.0.0.0:8000/topics/` for topic language analysis
6. Set up audio input recognizer, refers to [Mozilla DeepSpeech](https://github.com/mozilla/DeepSpeech)
    * Install by pip: `pip install deepspeech` or `pip install deepspeech-gpu` (Need to setup GPU supports);
    * Download [audio models](https://github.com/mozilla/DeepSpeech/releases) from Mozilla;
    * Add the path information to the [settings.init](./resources/settings.ini);
    * Note: all the audio input will be save temporarily under `./resources/audio_tmp/` 
    * The function won't enabled until all previous steps has been done correctly.
     
# Screenshots of demo
1. Only Content
<img src="./resources/screenshots/1.png?raw=true" alt="Content Mode" title="Optional Title" width="66%">
2. With context
<img src="./resources/screenshots/2.png?raw=true" alt="Contextual Mode" title="Optional Title" width="66%">

# License
1. This project is licensed under the GPL License

# Contact
This author.
