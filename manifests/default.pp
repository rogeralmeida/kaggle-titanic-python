exec { "apt-get update":
    command => "/usr/bin/apt-get update",
    onlyif => "/bin/sh -c '[ ! -f /var/cache/apt/pkgcache.bin ] || /usr/bin/find /etc/apt/* -cnewer /var/cache/apt/pkgcache.bin | /bin/grep . > /dev/null'",
}
$dependecies=["ipython-notebook", "python-numpy", "python-scipy", "python-matplotlib", "xclip", "wget",
"python-pandas", "python-sympy", "python-nose", "python-dev", "sqlite3", "libsqlite3-dev", "git-core", "python-mlpy"]

package {$dependecies:
	ensure => 'installed',
    require  => Exec['apt-get update'],
}

exec { "install easy_install":
    command => "wget https://bitbucket.org/pypa/setuptools/raw/bootstrap/ez_setup.py -O - | python",
    path => "/usr/bin/",
    creates => "/usr/bin/easy_install",
    require  => Package["python-dev"],
}
