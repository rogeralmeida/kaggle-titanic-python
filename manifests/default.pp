exec { "apt-get update":
    command => "/usr/bin/apt-get update",
    onlyif => "/bin/sh -c '[ ! -f /var/cache/apt/pkgcache.bin ] || /usr/bin/find /etc/apt/* -cnewer /var/cache/apt/pkgcache.bin | /bin/grep . > /dev/null'",
}

package { "ipython-notebook":
    ensure => "installed",
    require  => Exec['apt-get update'],
}

package { "python-dev":
    ensure => "installed",
    require  => Exec['apt-get update'],
}

package { "sqlite3":
    ensure => "installed",
    require => Exec["apt-get update"],
}

package { "libsqlite3-dev":
    ensure => "installed",
    require => Exec["apt-get update"],
}
package { "xclip":
    ensure => "installed",
    require  => Exec['apt-get update'],
}

package { "git-core":
	ensure => "installed",
}

package {'wget':
	ensure => 'installed',
	require => Package['python-dev'],
}

exec { "install easy_install":
    command => "wget https://bitbucket.org/pypa/setuptools/raw/bootstrap/ez_setup.py -O - | python",
    path => "/usr/bin/",
    creates => "/usr/bin/easy_install",
    require  => Package["python-dev"],
}
