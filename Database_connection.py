import pandas as pd
import numpy as np


import sys
import pandas as pd
import numpy as np

import cx_Oracle

def get_credentials(cred_file="credentials"):
    """Retrieves credentials (username, password, etc) from adequate file.

    Parameters
    ----------
    cred_file : str
        Name of the file in which to find the credentials to use, in order
        to establish a connection with the database.
        File should be of the form:
        '''
        username=[username]
        password=[password]
        host=[host]
        port=[port]
        sid=[sid]
        '''

    Returns
    -------
    dict
        Dictionary containing the credential information to connect to the db.

    """
    res = dict()
    with open(cred_file, 'r') as cred:
        for l in cred:
            s = l.split('=')
            res[s[0].strip()] = s[1].strip()
    return res


class Querier(object):
    def __init__(self, cred_file):
        """Creates a Querier object.

        -- self.__credentials are the credentials used to connect to the database
        -- self.__connection_str is used to connect to the db
        -- self.__connection is the connection object that ensures the link with \
            the db

        """
        self.__credentials = get_credentials(cred_file)
        self.__connection_str = u"{username}/{password}@{host}:{port}/?service_name={dsn}".format(
            **self.__credentials)
        self.__connection_dsn = cx_Oracle.makedsn(self.__credentials['host'],
                                                self.__credentials['port'],
                                                service_name=self.__credentials['dsn'])
        self.__connection = None

    def connect(self):
        try:
            # self.__connection = cx_Oracle.connect(self.__connection_str)
            self.__connection = cx_Oracle.connect(user=self.__credentials['username'],
                                                password=self.__credentials['password'],
                                                dsn=self.__connection_dsn)
        except cx_Oracle.DatabaseError as err:
            print("Problem with the database. \
Usually resolved by:\n\t* checking VPN connection\n\t* checking credential file\n\n{}".format(err))
            sys.exit(1)

    def disconnect(self):
        self.__connection.close()
        self.__connection = None

    def query(self, query):
        """Queries the database with the given query.

        Parameters
        ----------
        query : str
            Oracle SQL query.

        Returns
        -------
        pandas.DataFrame
            Result of pd.read_sql(query, connection), ie result of the asked \
                query

        If the connection has not yet been established, it is done automatically.
        """
        if self.__connection is None:
            self.connect()
        return pd.read_sql(query, self.__connection)

    def query_from_file(self, filename, i = 0):
        """Queries the database with the ith query in the given file.

        Parameters
        ----------
        filename : str
            Name of the file to fetch the query from.
        i : int
            Index of the query we want to run.

        Returns
        -------
        pandas.DataFrame
            Result of self.query(q) where q is the ith query in the given file.

        The ith query is defined by
            - the chunk of the file before the ith ';' line and after the (i-1)th,
            - or by the last query if there is too few (i=-1 gives the last query automatically)
            - or by the whole file if it contains no ';' line.
        """
        if self.__connection is None:
            self.connect()
        q = ""
        count = 0
        with open(filename, 'r') as f:
            for l in f:
                if l.strip() == ';' and count == i:
                    return self.query(q)
                elif l.strip() == ';':
                    count += 1
                    q = ""
                else:
                    q += l
        return self.query(q)
