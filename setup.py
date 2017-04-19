from setuptools import setup

setup(name='DeepChatModels',
        version='0.1',
        description='Conversation Models in TensorFlow',
        url='http://github.com/mckinziebrandon/DeepChatModels',
        author_email='mckinziebrandon@berkeley.edu',
        license='MIT',
        install_requires=[
            'numpy',
            'matplotlib',
            'pandas',
            'pyyaml',
            'git-code-debt',
            ],
        #extras_require={'tensorflow': ['tensorflow'],
        #    'tensorflow with gpu': ['tensorflow-gpu']},
        zip_safe=False)
