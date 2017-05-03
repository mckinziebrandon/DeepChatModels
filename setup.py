from setuptools import setup

setup(name='DeepChatModels',
        description='Conversation Models in TensorFlow',
        url='http://github.com/mckinziebrandon/DeepChatModels',
        author_email='mckinziebrandon@berkeley.edu',
        license='MIT',
        install_requires=[
            'numpy',
            'matplotlib',
            'pandas',
            'pyyaml',
            ],
        extras_require={'tensorflow': ['tensorflow'],
            'tensorflow gpu': ['tensorflow-gpu']},
        zip_safe=False)
