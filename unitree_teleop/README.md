# Teleoperation for Unitree Humanoid Robot


# 1. 📦 Prerequisites

We tested our code on Ubuntu 20.04 and Ubuntu 22.04, other operating systems may be configured differently.  

For more information, you can refer to [Official Documentation ](https://support.unitree.com/home/zh/Teleoperation) and [OpenTeleVision](https://github.com/OpenTeleVision/TeleVision).

## 1.1 🦾  inverse kinematics 

```bash
unitree@Host:~$ conda create -n tv python=3.8
unitree@Host:~$ conda activate tv
# If you use `pip install`, Make sure pinocchio version is 3.1.0
(tv) unitree@Host:~$ conda install pinocchio -c conda-forge
(tv) unitree@Host:~$ pip install meshcat
(tv) unitree@Host:~$ pip install casadi
```

## 1.2 🕹️ unitree_sdk2_python

```bash
# Install unitree_sdk2_python.
(tv) unitree@Host:~$ git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
(tv) unitree@Host:~$ cd unitree_sdk2_python
(tv) unitree@Host:~$ pip install -e .
```

# 2. ⚙️ Configuration

## 2.1 📥 basic

```bash
(tv) unitree@Host:~$ cd robot/avp_teleoperate
(tv) unitree@Host:~$ pip install -r requirements.txt
```

## 2.2 🔌 Local streaming

**2.2.1 Apple Vision Pro** 

Apple does not allow WebXR on non-https connections. To test the application locally, we need to create a self-signed certificate and install it on the client. You need a ubuntu machine and a router. Connect the Apple Vision Pro and the ubuntu **Host machine** to the same router.

1. install mkcert: https://github.com/FiloSottile/mkcert
2. check **Host machine** local ip address:

```bash
(tv) unitree@Host:~/avp_teleoperate$ ifconfig | grep inet
```

Suppose the local ip address of the **Host machine** is `192.168.123.2`

> p.s. You can use `ifconfig` command to check your **Host machine** ip address.

3. create certificate:

```bash
(tv) unitree@Host:~/avp_teleoperate$ mkcert -install && mkcert -cert-file cert.pem -key-file key.pem 192.168.123.2 localhost 127.0.0.1
```

place the generated `cert.pem` and `key.pem` files in `teleop`

```bash
(tv) unitree@Host:~/avp_teleoperate$ cp cert.pem key.pem ~/avp_teleoperate/teleop/
```

4. open firewall on server:

```bash
(tv) unitree@Host:~/avp_teleoperate$ sudo ufw allow 8012
```

5. install ca-certificates on Apple Vision Pro:

```bash
(tv) unitree@Host:~/avp_teleoperate$ mkcert -CAROOT
```

Copy the `rootCA.pem` via AirDrop to Apple Vision Pro and install it.

Settings > General > About > Certificate Trust Settings. Under "Enable full trust for root certificates", turn on trust for the certificate.

> In the new version of Vision OS 2, this step is different: After copying the certificate to the Apple Vision Pro device via AirDrop, a certificate-related information section will appear below the account bar in the top left corner of the Settings app. Tap it to enable trust for the certificate.

Settings > Apps > Safari > Advanced > Feature Flags > Enable WebXR Related Features.