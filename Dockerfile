  
# Ubuntu base
FROM ubuntu:18.04

# Update Ubuntu

RUN apt-get update

# Then install python3 and pip3
RUN apt-get install -y python3 python3-pip

# Installing the required file
RUN pip install -r requirements.txt

# install git
RUN apt-get install git

# git clone repository
RUN git clone git@github.com:gmortuza/vsm.git

CMD [ "python", "vsm/src/vsm.py" ]
