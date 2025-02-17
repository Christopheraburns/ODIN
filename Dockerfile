FROM amazonlinux:2

# Specify encoding
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install python-pip
RUN yum update -y && yum install -y python3.6 python3-pip && yum install -y tar && yum install -y xz && yum install -y libXtst libXxf86vm libXfixes libXrender libGL libSM 

RUN pip3 install opencv-python && pip3 install boto3 && pip3 install --upgrade Pillow && pip3 install --upgrade matplotlib

# Install flask server
RUN pip3 install -U Flask joblib sklearn;

COPY blender-2.82a-linux64.tar.xz /blender-2.82a-linux64.tar.xz

RUN tar xvf blender-2.82a-linux64.tar.xz

RUN cd blender-2.82a-linux64/2.82/python/bin && ./python3.7m -m ensurepip && ./python3.7m pip3 install opencv-python && ./python3.7m pip3 install --upgrade Pillow && ./python3.7m pip3 install --upgrade boto3 

COPY *.py /


COPY serve /opt/program/serve
RUN chmod 755 /opt/program/serve
ENV PATH=/opt/program:${PATH}echo

ENTRYPOINT ["/opt/program/serve"]
CMD serve