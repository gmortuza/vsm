  
# Ubuntu base
FROM ubuntu:18.04

# Update Ubuntu

RUN apt-get update

# Then install python3 and pip3
RUN apt-get install -y python3 python3-pip

# install git
RUN apt-get install -y git

# RUN useradd -ms /bin/bash jupyter

#vUSER jupyter
RUN git clone https://github.com/gmortuza/vsm.git

RUN pip3 install -r vsm/requirements.txt
# git clone repository


# Installing the required file

RUN python3 vsm/src/vsm.py

CMD ["jupyter", "notebook", "--ip=*"]
