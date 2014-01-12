# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  # Every Vagrant virtual environment requires a box to build off of.
  #config.vm.box = "precise64"

  # The url from where the 'config.vm.box' box will be fetched if it
  # doesn't already exist on the user's system.
  #config.vm.box_url = "http://files.vagrantup.com/precise64.box"
  config.vm.box = "Ubuntu 12.10 Quantal x86_64"
  config.vm.box_url = "https://github.com/downloads/roderik/VagrantQuantal64Box/quantal64.box"

  # Forward a port from the guest to the host, which allows for outside
  # computers to access the VM, whereas host only networking does not.
  #config.vm.forward_port 80, 8080
  # config.vm.network :forwarded_port, guest: 8888, host: 8888
  #config.vm.forward_port 8000, 8001
  #config.vm.forward_port 9999, 9998

  # config.vm.network :public_network, :bridge => 'en1: Wi-Fi (AirPort)'

  config.vm.provision "puppet" do |puppet|
    puppet.manifests_path = "manifests"
    puppet.manifest_file = "default.pp"
    puppet.options = "--verbose"
  end

end