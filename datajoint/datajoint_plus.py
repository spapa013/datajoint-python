"""
Classes and methods for datajoint_plus, an extension to DataJoint. 

The primary use case of datajoint_plus is to enable automatic hashing in DataJoint tables and to provide features to enhance the DataJoint master-part relationship. 

datajoint_plus disables table modification from virtual modules. 
"""

from os import confstr_names, error
import datajoint as dj
from datajoint.expression import QueryExpression
import inspect
import collections
import pandas as pd
import hashlib
import simplejson
import re
from enum import Enum
import numpy as np
from collections import Counter
from IPython.display import display, clear_output
from ipywidgets.widgets import Output, HBox, Label
import traceback
import copy
import logging
from .table import FreeTable
from .user_tables import UserTable

__version__ = "0.0.17"


class classproperty:
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


class ValidationError(dj.DataJointError):
    pass


class OverwriteError(dj.DataJointError):
    pass


def generate_hash(rows, add_constant_columns:dict=None):
    """
    Generates hash for provided rows. 

    :param rows (pd.DataFrame, dict): Rows to hash. `type(rows)` must be able to instantiate a pandas dataframe.
    :param add_constant_columns (dict):  Each key:value pair will be passed to the dataframe to be hashed as `df[k]=v`, adding a column length `len(df)` with name `k` and filled with values `v`. 

    :returns: md5 hash
    """
    df = pd.DataFrame(rows)
    if add_constant_columns is not None:
        assert isinstance(add_constant_columns, dict), f' arg add_constant_columns must be Python dictionary instance.'
        for k, v in add_constant_columns.items():
            df[k] = v
    df = df.sort_index(axis=1).sort_values(by=[*df.columns])
    encoded = simplejson.dumps(df.to_dict(orient='records')).encode()
    dhash = hashlib.md5()
    dhash.update(encoded)
    return dhash.hexdigest()


def _validate_rows_for_hashing(rows):
    """
    Validates rows for `generate_hash`.
    """
    validated = False
    if isinstance(rows, pd.DataFrame):
        pass
    elif (inspect.isclass(rows) and issubclass(rows, QueryExpression)) or isinstance(rows, QueryExpression):
        pass
    elif isinstance(rows, list) or isinstance(rows, tuple):
        for row in rows:
            assert isinstance(row, collections.abc.Mapping), 'Cannot hash attributes unless row attributes are named. Try inserting a pandas dataframe, a DataJoint expression or a list of dictionaries.'
        pass
    else:
        raise ValidationError('Format of rows not recognized. Try inserting a list of dictionaries, a DataJoint expression or a pandas dataframe.')

        
def validate_and_generate_hash(rows, **kwargs):
    """
    Generates hash for provided rows with row validation for DataJoint.

    :rows: see `generate_hash`
    :kwargs: passed to `generate_hash`

    :returns: hash from `generate_hash`
    """
    _validate_rows_for_hashing(rows)
    return generate_hash(rows, **kwargs)
    

def split_full_table_name(full_table_name:str):
    """
    Splits full_table_name from DataJoint tables and returns a tuple of (database, table_name).

    :param (str): full_table_name from DataJoint tables
    
    :returns (tuple): (database, table_name)
    """
    return tuple(s.strip('`') for s in full_table_name.split('.'))


def reform_full_table_name(schema_name:str, table_name:str):
    """
    Reforms full_table_name from DataJoint schema name and a table_name.

    :param schema_name (str): name of schema
    :param table_name (str): name of table
    
    :returns: full_table_name
    """
    return '.'.join(['`'+schema_name+'`', '`'+table_name+'`'])


def format_table_name(table_name, snake_case=False, part=False):
    """
    Splits full_table_name from DataJoint tables and returns a tuple of (database, table_name).

    :param (str): full_table_name from DataJoint tables
    
    :returns (tuple): (database, table_name)
    """
    if not snake_case:
        if not part:
            return table_name.title().replace('_','').replace('#','')
        else:
            return table_name.title().replace('__','.').replace('_','').replace('#','')
    else:
        if not part:
            return table_name.lower().strip('_').replace('#','')
        else:
            return table_name.lower().replace('__','.').strip('_').replace('#','')


def parse_definition(definition):
    """
    Parses DataJoint definitions. Extracts the following line types, where lines are separated by newlines `\\n`:
        - headers: a line that starts with hash `#`.
        - dependencies: a line that does not start with hash `#` and contains an arrow `->`.
        - attributes: a line that contains a colon `:`, and may contain a hash `#`, as long as the colon `:` precedes the hash `#`.
        - divider: a line that does not start with a hash `#` and contains a DataJoint divider `---`.
        - non-matches: a line that does not meet any of the above criteria. 
        
    :param definition: DataJoint table definition to be parsed.

    :returns: 
        parsed_inds (dict): dictionary of line types and matching line indexes. 
        parsed_contents (dict): a dictionary of line types and matching line contents.
        parsed_stats (dict): a dictionary of line types and a summary of the number of line matches for each type. 
    """

    lines = re.split('\n', definition) # seperate definition lines
    lines_without_spaces = [re.sub(' +', '', l) for l in lines]

    char_inds = []
    for l in lines_without_spaces:
        ind_dict = {}
        for name, char in zip(['hash', 'divider', 'arrow', 'colon'], ['#', '---', '->', ':']):
            ind_dict[name] = [m.start() for m in re.finditer(char, l)]
        char_inds.append(ind_dict)

    df = pd.DataFrame(char_inds)

    df['contains_hash'] = df.apply(lambda x: x.hash != [], axis=1)
    df['contains_divider'] = df.apply(lambda x: x.divider != [], axis=1)
    df['contains_arrow'] = df.apply(lambda x: x.arrow != [], axis=1)
    df['contains_colon'] = df.apply(lambda x: x.colon != [], axis=1)
    df['hash_pos_0'] = df.apply(lambda x: 0 in x.hash, axis=1)
    df['colon_before_hash'] = df.apply(lambda x: x.colon[0] < x.hash[0] if x.colon and x.hash else False, axis=1)

    header_query = "hash_pos_0 == True"
    dependency_query = "hash_pos_0 == False and contains_arrow == True"
    attribute_query = "(contains_colon==True and contains_hash==False) or colon_before_hash==True"
    divider_query = "hash_pos_0 == False and contains_divider == True"

    names = ['headers', 'dependencies', 'attributes', 'dividers']
    queries = [header_query, dependency_query, attribute_query, divider_query]

    parsed_inds = {}
    for name, query in zip(names, queries):
        q = df.query(query)
        df = df.drop(df.query(query).index.values)
        if len(q)> 0:
            parsed_inds[name] = q.index.values
        else:
            parsed_inds[name] = []
    parsed_inds['non-matches'] = df.index.values

    parsed_contents = {}
    parsed_stats = {}

    for n in names + ['non-matches']:
        parsed_contents[n] = [lines[i] for i in parsed_inds[n]]
        parsed_stats[n] = f'Found {len(parsed_inds[n])} line(s) matching the profile of {n}.'

    return parsed_inds, parsed_contents, parsed_stats


def reform_definition(parsed_inds, parsed_contents):
    """
    Reforms DataJoint definition after parsing by `parse_definition`. 

    :param parsed_inds (dict): see `parse_definition`
    :param parsed_contents (dict): see `parse_definition`

    :returns: DataJoint definition
    """
    n_lines = len(np.concatenate([i for i in parsed_inds.values()]))
    content_list = [''] * n_lines
    for ii, cc in zip(parsed_inds.values(), parsed_contents.values()):
        for i, c in zip(ii, cc):
            content_list[int(i)] = c

    definition = """"""
    for c in content_list:
        definition += c + """\n"""
    
    return definition


def _is_overwrite_validated(attr, group, overwrite_rows):
    """
    Checks if attr is in group and is overwriteable.
    """
    assert isinstance(overwrite_rows, bool), 'overwrite_rows must be a boolean.'
    if not overwrite_rows:
        if attr in group:
            raise OverwriteError(f'Attribute "{attr}" already in rows. To overwrite, set overwrite_rows=True.')
        else:
            return True
    return True


class JoinMethod(Enum):
    PRIMARY = 'primary_only'
    SECONDARY = 'rename_secondaries'
    COLLISIONS = 'rename_collisions'
    ALL = 'rename_all'


class Base:
    _is_insert_validated = False
    _enable_table_modification = True

    # required for hashing
    enable_hashing = False
    hash_name = None
    hashed_attrs = None

    # hash params
    hash_group = False
    hash_table_name = False
    _hash_len = None

    # header params
    _add_hash_name_to_header = True
    _add_hashed_attrs_to_header = True
    _add_hash_params_to_header = True
    
    @classmethod
    def init_validation(cls, **kwargs):
        """
        Validation for initialization of subclasses of abstract class Base. 
        """
        for attr in ['enable_hashing', 'hash_group', 'hash_table_name', '_add_hash_name_to_header', '_add_hash_params_to_header', '_add_hashed_attrs_to_header']:
            assert isinstance(getattr(cls, attr), bool), f'"{attr}" must be boolean.'           

        for attr in ['hash_name', 'hashed_attrs']:
            assert not isinstance(getattr(cls, attr), bool), f'"{attr}" must not be boolean.'
        
        if cls.enable_hashing:
            for required in ['hash_name', 'hashed_attrs']:
                if getattr(cls, required) is None:
                    raise NotImplementedError(f'Hashing requires class to implement the property "{required}".')
        
        # ensure one attribute for "hash_name"
        if cls.hash_name is not None:
            if isinstance(cls.hash_name, list) or isinstance(cls.hash_name, tuple):
                if len(cls.hash_name) > 1:
                    raise NotImplementedError(f'Only one attribute allowed in "hash_name".')
                else:
                    cls.hash_name = cls.hash_name[0]

        # ensure "hashed_attrs" wrapped in list or tuple
        if cls.hashed_attrs is not None:
            if not isinstance(cls.hashed_attrs, list) and not isinstance(cls.hashed_attrs, tuple):
                cls.hashed_attrs = [cls.hashed_attrs]
            else:
                cls.hashed_attrs = cls.hashed_attrs

        # ensure hash_name and hashed_attrs are disjoint
        if cls.hash_name is not None and cls.hashed_attrs is not None:
            if not set((cls.hash_name,)).isdisjoint(cls.hashed_attrs):
                raise NotImplementedError(f'attributes in "hash_name" and "hashed_attrs" must be disjoint.')

        # modify header
        cls._add_hash_info_to_header(
            add_hash_name=cls.hash_name is not None and cls._add_hash_name_to_header, 
            add_hashed_attrs=cls.hashed_attrs is not None and cls._add_hashed_attrs_to_header,
            add_hash_group=cls.hash_group and cls._add_hash_params_to_header,
            add_hash_table_name=cls.hash_table_name and cls._add_hash_params_to_header,
            add_hash_part_table_names='hash_part_table_names' in kwargs and kwargs['hash_part_table_names'] and cls._add_hash_params_to_header,
        )

    @classproperty
    def hash_len(cls):
        if cls.hash_name is not None and cls._hash_len is None:
            cls._hash_name_validation()
        return cls._hash_len

    @classmethod
    def insert_validation(cls):
        """
        Validation for insertion to DataJoint tables that are subclasses of abstract class Base. 
        """
        # ensure hash_name and hashed_attrs are disjoint
        if cls.hash_name is not None and cls.hashed_attrs is not None:
            if not set((cls.hash_name,)).isdisjoint(cls.hashed_attrs):
                raise NotImplementedError(f'attributes in "hash_name" and "hashed_attrs" must be disjoint.')

        cls._is_insert_validated = True

    @classmethod
    def load_dependencies(cls, force=False):
        """
        Loads dependencies into DataJoint networkx graph. 
        """
        load = False
        if not force:
            if not cls.connection.dependencies._loaded:
                load = True
        else:
            load = True
        if load:
            output = Output()
            display(output)
            with output:
                pending_text = Label('Loading schema dependencies...')
                confirmation = Label('Success.')
                confirmation.layout.display = 'none'
                display(HBox([pending_text, confirmation]))
                cls.connection.dependencies.load()
                confirmation.layout.display = None

    @classmethod
    def add_constant_attrs_to_rows(cls, rows, constant_attrs:dict={}, overwrite_rows=False):
        """
        Adds attributes to all rows.

        :param rows (pd.DataFrame, QueryExpression, list, tuple): rows to pass to DataJoint `insert`. 
        :param constant_attrs (dict): Python dictionary to add to every row in rows
        :overwrite_rows (bool): Whether to overwrite key/ values in rows. If False, conflicting keys will raise a ValidationError.

        :returns: modified rows
        """   
        assert isinstance(constant_attrs, dict), 'arg "constant_attrs" must be a Python dictionary.'
        
        if constant_attrs != {}:
            if isinstance(rows, pd.DataFrame):
                rows = copy.deepcopy(rows)

                for k, v in constant_attrs.items():
                    if _is_overwrite_validated(k, rows, overwrite_rows):
                        rows[k] = v

            elif (inspect.isclass(rows) and issubclass(rows, QueryExpression)) or isinstance(rows, QueryExpression): 
                rows = rows.proj(..., **{k : f"'{v}'" for k, v in constant_attrs.items() if _is_overwrite_validated(k, rows.heading.names, overwrite_rows)})

            elif isinstance(rows, list) or isinstance(rows, tuple):
                rows = copy.deepcopy(rows)

                for row in rows:
                    assert isinstance(row, collections.abc.Mapping), 'Cannot hash attributes unless row attributes are named. Try inserting a pandas dataframe, a DataJoint expression or a list of dictionaries.'
                    for k, v in constant_attrs.items():
                        if _is_overwrite_validated(k, row.keys(), overwrite_rows):
                            row[k] = v
            else:
                raise ValidationError('Row format not recognized. Try providing a pandas dataframe, a DataJoint expression or a list of dictionaries.')

            return rows

    @classmethod
    def include_attrs(cls, *args):
        """
        Returns a projection of cls that includes only the attributes passed as args.

        Note: The projection is NOT guaranteed to have unique rows, even if it contains only primary keys. 
        """
        return cls.proj(..., **{a: '""' for a in cls.heading.names if a not in args}).proj(*[a for a in cls.heading.names if a in args])
    
    @classmethod
    def exclude_attrs(cls, *args):
        """
        Returns a projection of cls that excludes all attributes passed as args. 
        
        Note: The projection is NOT guaranteed to have unique rows, even if it contains only primary keys. 
        """
        return cls.proj(..., **{a: '""' for a in cls.heading.names if a in args}).proj(*[a for a in cls.heading.names if a not in args])
          
    @classmethod
    def _parse_hash_name_attribute(cls, source='self'):
        """
        Computes length of hash_name attribute in DataJoint table heading.

        :param source (str): the source of DataJoint heading to check
            - self: checks own heading
            - master: checks master table heading (only for part tables)

        :returns: 
            - hash type (str)
            - hash len (int)
        """

        if source == 'self':
            attributes = cls.heading.attributes

        elif source == 'master':
            attributes = cls.master.heading.attributes
        
        else:
            raise ValueError(f'heading source: {source} not recognized.')

        hash_type, hash_len = re.findall('\w+', attributes[cls.hash_name].type)

        return hash_type, int(hash_len)
    
    @classmethod
    def hash1(cls, rows, **kwargs):
        """
        Hashes rows and requires a single hash as output.
        
        See `hash` for kwargs.
        
        :returns (str): hash
        """
        hashes = cls.hash(rows, **kwargs)
        assert len(hashes) == 1, 'Multiple hashes found. hash1 must return only 1 hash.'
        return hashes[0]

    @classmethod
    def hash(cls, rows, unique=False):
        """
        Hashes rows.
        
        :param rows: rows containing attributes to be hashed. 
        :unique: If True, only unique hashes will be returned. If False, all hashes returned. 
        
        returns (list): list with hash(es)
        """
        return cls.add_hash_to_rows(rows)[cls.hash_name].unique().tolist() if unique else cls.add_hash_to_rows(rows)[cls.hash_name].tolist()

    @classmethod
    def restrict_with_hash(cls, hash, hash_name=None):
        """
        Returns table restricted with hash. 

        :param hash: hash to restrict with
        :param hash_name: name of attribute that contains hashes. 
            If hash_name is not None:
                Will use hash_name instead of cls.hash_name
            If hash_name is None:
                Will use cls.hash_name or raise ValidationError if cls.hash_name is None

        :returns: Table restricted with hash. 
        """
        if hash_name is None and hasattr(cls, 'hash_name'):
            hash_name = cls.hash_name
        
        if hash_name is None:
            raise ValidationError('Table does not have "hash_name" defined, provide it to restrict with hash.')
            
        return cls & {cls.hash_name: hash}

    @classmethod
    def _add_hash_info_to_header(cls, add_hash_name=False, add_hashed_attrs=False, add_hash_group=False, add_hash_table_name=False, add_hash_part_table_names=False):
        """
        Modifies definition header to include hash_name and hashed_attrs with a parseable syntax. 

        :param add_hash_name (bool): Whether to add hash_name to header
        :param add_hashed_attrs (bool): Whether to add hashed_attrs to header
        """
        if hasattr(cls, 'definition') and isinstance(cls.definition, str):
            inds, contents, _ = parse_definition(cls.definition)
            headers = contents['headers']

            if len(headers) >= 1:
                header = headers[0]

            else:
                # create header
                header = """#"""

            # append hash info to header
            if add_hash_name:
                header += f" | hash_name = {cls.hash_name};" 
            
            if add_hash_group:
                header += f" | hash_group = True;" 
            
            if add_hash_table_name:
                header += f" | hash_table_name = True;" 
            
            if add_hash_part_table_names:
                header += f" | hash_part_table_names = True;" 

            if add_hashed_attrs:
                header += f" | hashed_attrs = "
                for i, h in enumerate(cls.hashed_attrs):
                    header += f"{h}, " if i+1 < len(cls.hashed_attrs) else f"{h};"
            
            try:
                # replace existing header with modified header
                contents['headers'][0] = header
        
            except IndexError:
                # add header
                contents['headers'].extend([header])
                
                # header should go before any dependencies or attributes
                header_ind = np.min(np.concatenate([inds['dependencies'], inds['attributes']]))
                inds['headers'].extend([header_ind])
                
                # slide index over 1 to accommodate new header
                for n in [k for k in inds.keys() if k not in ['headers']]:
                    if len(inds[n])>0:
                        inds[n][inds[n] >= header_ind] += 1
            
            # reform and set definition
            cls.definition = reform_definition(inds, contents)

    @classmethod
    def parse_hash_info_from_header(cls):
        """
        Parses hash_name and hashed_attrs from DataJoint table header and sets properties in class. 
        """
        header = cls.heading.table_info['comment']
        matches = re.findall(r'\|(.*?);', header)
        if matches:
            for match in matches:
                result = re.findall('\w+', match)

                if result[0] == 'hash_name':
                    cls.hash_name = result[1]

                if result[0] == 'hash_group':
                    if result[1] == 'True':
                        cls.hash_group = True

                if result[0] == 'hash_table_name':
                    if result[1] == 'True':
                        cls.hash_table_name = True

                if result[0] == 'hash_part_table_names':
                    if result[1] == 'True':
                        cls.hash_part_table_names = True

                if result[0] == 'hashed_attrs':
                    cls.hashed_attrs = result[1:]

    @classmethod
    def add_hash_to_rows(cls, rows, overwrite_rows=False):
        """
        Adds hash to rows.

        :param rows (pd.DataFrame, QueryExpression, list, tuple): rows to pass to DataJoint `insert`.
        :param hash_table_name (bool): Whether to include table_name in rows for hashing
        :overwrite_rows (bool): Whether to overwrite key/ values in rows. If False, conflicting keys will raise a ValidationError. 

        :returns: modified rows
        """
        assert cls.hashed_attrs is not None, 'Table must have hashed_attrs defined. Check if hashing was enabled for this table.'

        hash_table_name = True if cls.hash_table_name or (issubclass(cls, dj.Part) and hasattr(cls.master, 'hash_part_table_names') and getattr(cls.master, 'hash_part_table_names')) else False

        if hash_table_name:
            table_name = {'#__table_name__': cls.table_name}
        else:
            table_name = None
            
        if isinstance(rows, pd.DataFrame):
            rows = rows.copy()
        
        elif (inspect.isclass(rows) and issubclass(rows, QueryExpression)) or isinstance(rows, QueryExpression):
            rows = pd.DataFrame(rows.fetch())

        elif isinstance(rows, list) or isinstance(rows, tuple):
            rows = pd.DataFrame(rows)
        else:
            raise ValidationError('Format of rows not recognized. Try inserting a list of dictionaries, a DataJoint expression or a pandas dataframe.')

        for a in cls.hashed_attrs:
            assert a in rows.columns.values, f'hashed_attr "{a}" not in rows. Row names are: {rows.columns.values}'

        if _is_overwrite_validated(cls.hash_name, rows, overwrite_rows):
            rows_to_hash = rows[[*cls.hashed_attrs]]

            if cls.hash_group:
                rows[cls.hash_name] = generate_hash(rows_to_hash, add_constant_columns=table_name)[:cls.hash_len]

            else:
                rows[cls.hash_name] = [generate_hash([row], add_constant_columns=table_name)[:cls.hash_len] for row in rows_to_hash.to_dict(orient='records')]
                
        return rows

    @classmethod
    def _prepare_insert(cls, rows, constant_attrs, overwrite_rows=False, skip_hashing=False):
        """
        Prepares rows for insert by checking if table has been validated for insert, adds constant_attrs and performs hashing. 
        """
        
        if not cls._is_insert_validated:
            cls.insert_validation()
        
        if constant_attrs != {}:
            rows = cls.add_constant_attrs_to_rows(rows, constant_attrs, overwrite_rows)

        if cls.enable_hashing and not skip_hashing:
            try:
                rows = cls.add_hash_to_rows(rows, overwrite_rows=overwrite_rows)

            except OverwriteError as err:
                new = err.args[0]
                new += ' Or, to skip the hashing step, set skip_hashing=True.'
                raise OverwriteError(new) from None

        return rows


class MasterBase(Base):
    hash_part_table_names = False
    _is_hash_name_validated = False

    def __init_subclass__(cls, **kwargs):
        cls.init_validation()

    @classmethod
    def init_validation(cls):
        """
        Validation for initialization of subclasses of abstract class MasterBase. 
        """
        for attr in ['hash_table_name', 'hash_part_table_names']:
            assert isinstance(getattr(cls, attr), bool), f'"{attr}" must be a boolean.'

        super().init_validation(hash_table_name=cls.hash_table_name, hash_part_table_names=cls.hash_part_table_names)

    @classmethod
    def insert_validation(cls):
        """
        Validation for insertion into subclasses of abstract class MasterBase. 
        """
        if cls.hash_name is not None:
            if cls.hash_name not in cls.heading.names:
                raise ValidationError(f'Attribute "{cls.hash_name}" in property "hash_name" must be present in table heading.')

            # hash_name validation
            if not cls._is_hash_name_validated:
                cls._hash_name_validation()
        
        super().insert_validation()
    
    @classmethod
    def _hash_name_validation(cls):
        """
        Validates hash_name and sets hash_len
        """
        hash_type, hash_len = cls._parse_hash_name_attribute()

        if hash_type != 'varchar' or not (hash_len > 0 and hash_len <= 32):
            raise ValidationError(f'hash_name "{cls.hash_name}" must be a "varchar" type > 0 and <= 32 characters')
        
        cls._hash_len = hash_len

        cls._is_hash_name_validated = True

    @classmethod
    def parts(cls, as_objects=False, as_cls=False, reload_dependencies=False):
        """
        Wrapper around Datajoint function `parts` that enables returning parts as part_names, objects, or classes, and enables reloading of Datajoint networkx graph dependencies.

        :param as_objects: 
            If True, returns part tables as objects
            If False, returns part table names
        :param as_cls:
            If True, returns part table classes (will override as_objects)
        :param reload_dependencies: 
            If True, will force reload Datajoint networkx graph dependencies. 

        :returns: list
        """
        cls.load_dependencies(force=reload_dependencies)

        cls_parts = [getattr(cls, d) for d in dir(cls) if inspect.isclass(getattr(cls, d)) and issubclass(getattr(cls, d), dj.Part)]
        for cls_part in [p.full_table_name for p in cls_parts]:
            if cls_part not in super().parts(cls):
                logging.warning('Part table defined in class definition not found in DataJoint graph. Reload dependencies.')

        if not as_cls:
            return super().parts(cls, as_objects=as_objects)
        else:
            return cls_parts

    @classmethod
    def number_of_parts(cls, parts_kws={}):
        """
        Returns the number of part tables belonging to cls. 
        """
        return len(cls.parts(**parts_kws))

    @classmethod
    def has_parts(cls, parts_kws={}):
        """
        Returns True if cls has part tables. 
        """
        return cls.number_of_parts(parts_kws) > 0

    @classmethod
    def _format_parts(cls, parts):
        """
        Formats the part tables in arg parts. 
        """
        if not isinstance(parts, list) and not isinstance(parts, tuple):
            parts = [parts]
        
        new = []
        for part in parts:
            if inspect.isclass(part) and issubclass(part, dj.Part):
                new.append(part()) # instantiate if a class

            elif isinstance(part, dj.Part):
                new.append(part)

            elif isinstance(part, QueryExpression) and not isinstance(part, dj.Part):
                raise ValidationError(f'Arg "{part.full_table_name}" is not a valid part table.')

            else:
                raise ValidationError(f'Arg "{part}" must be a part table or a list or tuple containing one or more part tables.')

        return new

    @classmethod
    def union_parts(cls, part_restr={}, include_parts=None, exclude_parts=None, filter_out_len_zero=False, parts_kws={}):
        """
        Returns union of part table primary keys after optional restriction. Requires all part tables in union to have identical primary keys. 

        :params: see `restrict_parts`.

        :returns: numpy array object
        """  
        return np.sum([p.proj() for p in cls.restrict_parts(part_restr=part_restr, include_parts=include_parts, exclude_parts=exclude_parts, filter_out_len_zero=filter_out_len_zero, parts_kws=parts_kws)])

#     @classmethod
#     def keys_not_in_parts(cls, part_restr={}, include_parts=None, exclude_parts=None, master_restr={}, parts_kws={}):
#         return (cls & master_restr) - cls.union_parts(include_parts=include_parts, exclude_parts=exclude_parts, part_restr=part_restr, parts_kws=parts_kws)

    @classmethod
    def join_parts(cls, part_restr={}, join_method=None, join_with_master=False, include_parts=None, exclude_parts=None, filter_out_len_zero=False, parts_kws={}):
        """
        Returns join of part tables after optional restriction. 

        :params part_restr, include_parts, exclude_parts, parts_kws: see `restrict_parts`.
        :param join_method (str):
            - 'primary_only' - will project out secondary keys and will only join on primary keys
            - 'rename_secondaries' - will add the part table to all secondary keys
            - 'rename_collisions' - will add the part table name to secondary keys that are present in more than one part table
            - 'rename_all' - will add the part table name to all primary and secondary keys

        :param join_with_master (bool): If True, parts will be joined with cls before returning. 

        :returns: numpy array object
        """
        parts = cls.restrict_parts(part_restr=part_restr, include_parts=include_parts, exclude_parts=exclude_parts, filter_out_len_zero=filter_out_len_zero, parts_kws=parts_kws)
        
        if join_with_master:
            parts = [FreeTable(cls.connection, cls.full_table_name)] + parts

        collisions = None
        if join_method is None:
            try:
                return np.product(parts)

            except:
                traceback.print_exc()
                msg = 'Join unsuccessful. Try one of the following: join_method = '
                for i, j in enumerate(JoinMethod):
                    msg += f'"{j.value}", ' if i+1 < len(JoinMethod) else f'"{j.value}".'
                print(msg)
                return
        
        elif join_method == JoinMethod.PRIMARY.value:
            return np.product([p.proj() for p in parts])
        
        elif join_method == JoinMethod.SECONDARY.value:
            attributes_to_rename = [p.heading.secondary_attributes for p in parts]
            
        elif join_method == JoinMethod.COLLISIONS.value:
            attributes_to_rename = [p.heading.secondary_attributes for p in parts]
            collisions = [item for item, count in Counter(np.concatenate(attributes_to_rename)).items() if count > 1]
            
        elif join_method == JoinMethod.ALL.value:
            attributes_to_rename = [list(p.heading.attributes.keys()) for p in parts]

        else:
            msg = f'join_method "{join_method}" not implemented. Available methods: '
            for i, j in enumerate(JoinMethod):
                msg += f'"{j.value}", ' if i+1 < len(JoinMethod) else f'"{j.value}".'
            raise NotImplementedError(msg)

        renamed_parts = []
        for p, attrs in zip(parts, attributes_to_rename):
            if isinstance(p, dj.Part):
                name = format_table_name(p.table_name, snake_case=True, part=True).split('.')[1]
            else:
                name = format_table_name(p.table_name, snake_case=True)

            if collisions is not None:
                renamed_attribute = {name + '_' + a : a for a in attrs if a in collisions}
            else:
                renamed_attribute = {name + '_' + a : a for a in attrs}
            renamed_parts.append(p.proj(..., **renamed_attribute))
            
        return np.product(renamed_parts)
    
    @classmethod
    def restrict_parts(cls, part_restr={}, include_parts=None, exclude_parts=None, filter_out_len_zero=False, parts_kws={}):
        """
        Restricts part tables of cls. 

        :param part_restr: restriction to restrict part tables with.
        :param include_parts (part table or list of part tables): part table(s) to restrict. If None, will restrict all part tables of cls.
        :param exclude_parts (part table or list of part tables): part table(s) to exclude from restriction
        :param parts_kws (dict): kwargs to pass to cls.parts. If no kwargs are provided, `as_cls=True` will be passed to cls.parts.
        :param filter_out_len_zero (bool): If True, parts with len = 0 after restriction are excluded from list.
        """
        assert cls.has_parts(parts_kws), 'No part tables found.'
        parts_kws = {k:v for k,v in parts_kws.items() if k not in ['reload_dependencies']}

        if include_parts is None:
            parts = cls.parts(**parts_kws) if parts_kws!={} else cls.parts(as_cls=True)
        
        else:
            parts = cls._format_parts(include_parts)
        
        if exclude_parts is not None:
            parts = [p for p in parts if p.full_table_name not in [e.full_table_name for e in cls._format_parts(exclude_parts)]]
        
        parts = [p & part_restr for p in parts]

        return  parts if not filter_out_len_zero else [p for p in parts if len(p)>0]
    
    @classmethod
    def restrict_one_part(cls, part_restr={}, include_parts=None, exclude_parts=None, filter_out_len_zero=True, parts_kws={}):
        """
        Calls `restrict_parts` with filter_out_len_zero=True by default. If not exactly one part table is returned, then a ValidationError will be raised.

        WARNING: If the attributes in part and part_restr are mutually exclusive, then len(part & part_restr) > 0. 
        This means that if a part table and part_restr don't share any column names, then the part table will not be filtered out from `restrict_parts` even if the part_restr has no matching entries in that part table.

        :params: see `restrict_parts`.

        :returns: part table after restriction.
        """
        parts = cls.restrict_parts(part_restr=part_restr, include_parts=include_parts, exclude_parts=exclude_parts, filter_out_len_zero=filter_out_len_zero, parts_kws=parts_kws)

        if len(parts) > 1:
            raise ValidationError('part_restr can restrict multiple part tables.')
        
        elif len(parts) < 1:
            raise ValidationError('part_restr can not restrict any part tables.')
        
        else:
            return parts[0]

    r1p = restrict_one_part # alias for restrict_one_part

    @classmethod
    def part_table_names_with_hash(cls, hash, hash_name=None, include_parts=None, exclude_parts=None, filter_out_len_zero=True, parts_kws={}):
        """
        Calls `restrict_parts_with_hash` with filter_out_len_zero=True by default.

        :params: see `restrict_parts_with_hash`

        :returns: list of part table names that contain hash.
        """
        parts = cls.restrict_parts_with_hash(hash=hash, hash_name=hash_name, include_parts=include_parts, exclude_parts=exclude_parts, filter_out_len_zero=filter_out_len_zero, parts_kws=parts_kws)
        return [format_table_name(r.table_name, part=True) for r in parts]

    @classmethod
    def restrict_one_part_with_hash(cls, hash, hash_name=None, include_parts=None, exclude_parts=None, filter_out_len_zero=True, parts_kws={}):
        """
        Calls `restrict_parts_with_hash` with filter_out_len_zero=True by default. If not exactly one part table is returned, then a ValidationError will be raised.

        :params: see `restrict_parts_with_hash`

        :returns: part table after restriction
        """
        parts = cls.restrict_parts_with_hash(hash=hash, hash_name=hash_name, include_parts=include_parts, exclude_parts=exclude_parts, filter_out_len_zero=filter_out_len_zero, parts_kws=parts_kws)

        if len(parts) > 1:
            raise ValidationError('Hash found in multiple part tables.')
        
        elif len(parts) < 1:
            raise ValidationError('Hash not found in any part tables.')
        
        else:
            return parts[0]
    
    r1pwh = restrict_one_part_with_hash # alias for restrict_one_part_with_hash

    @classmethod
    def restrict_parts_with_hash(cls, hash, hash_name=None, include_parts=None, exclude_parts=None, filter_out_len_zero=False, parts_kws={}):
        """
        Checks all part tables and returns the part table that is successfully restricted by {'hash_name': hash}. 

        Note: If hash_name is not provided, cls.hash_name will be tried. 

        A successful restriction is defined by:
            - 'hash_name' is in the part table heading
            - len(part & {'hash_name': hash}) > 0 if filter_out_len_zero=True

        :param hash: hash to restrict with
        :param hash_name: name of attribute that contains hash. If hash_name is None, cls.hash_name will be used.
        :params include_parts, exclude_parts, parts_kws: see `restrict_parts`

        :returns: list of part tables after restriction
        """  
        if hash_name is None and hasattr(cls, 'hash_name'):
            hash_name = cls.hash_name

        if hash_name is None:
            raise ValidationError('Table does not have "hash_name" defined, provide it to restrict with hash.')
        
        parts = cls.restrict_parts(part_restr={hash_name: hash}, include_parts=include_parts, exclude_parts=exclude_parts, filter_out_len_zero=filter_out_len_zero, parts_kws=parts_kws)

        return [p for p in parts if hash_name in p.heading.names]
    
    @classmethod
    def hashes_not_in_parts(cls, hash_name=None, part_restr={}, include_parts=None, exclude_parts=None, filter_out_len_zero=False, parts_kws={}):
        """
        Restricts master table to any hashes not found in any of its part tables.

        :param hash_name: name of attribute that contains hash. If hash_name is None, cls.hash_name will be used.
        :params part_restr, include_parts, exclude_parts, parts_kws: see `restrict_parts`

        :returns: cls after restriction
        """
        if hash_name is None and hasattr(cls, 'hash_name'):
            hash_name = cls.hash_name

        if hash_name is None:
            raise ValidationError('Table does not have "hash_name" defined, provide it to restrict with hash.')

        return cls - np.sum([(dj.U(cls.hash_name) & p) for p in cls.restrict_parts(part_restr=part_restr, include_parts=include_parts, exclude_parts=exclude_parts, filter_out_len_zero=filter_out_len_zero, parts_kws=parts_kws)])

    @classmethod
    def insert(cls, rows, replace=False, skip_duplicates=False, ignore_extra_fields=False, allow_direct_insert=None, reload_dependencies=False, insert_to_parts=None, insert_to_parts_kws={}, skip_hashing=False, constant_attrs={}, overwrite_rows=False):
        """
        Insert rows to cls.

        :params rows, replace, skip_duplicates, ignore_extra_fields, allow_direct_insert: see DataJoint insert function.
        :param reload_dependencies (bool): force reload DataJoint networkx graph dependencies before insert.
        :param insert_to_parts (part table or list of part tables): part table(s) to insert to after master table insert.
        :param insert_to_parts_kws (dict): kwargs to pass to part table insert function.
        :param skip_hashing (bool): If True, hashing will be skipped if hashing is enabled. 
        :param constant_attrs (dict): Python dictionary to add to every row of rows
        :overwrite_rows (bool): Whether to overwrite key/ values in rows. If False, conflicting keys will raise a ValidationError.
        """
        if not cls._is_insert_validated:
            cls.insert_validation()
        
        if insert_to_parts is not None:
            assert cls.has_parts(parts_kws=dict(reload_dependencies=reload_dependencies)), 'No part tables found.'
            insert_to_parts = cls._format_parts(insert_to_parts)

        rows = cls._prepare_insert(rows, constant_attrs=constant_attrs, overwrite_rows=overwrite_rows, skip_hashing=skip_hashing)

        super().insert(cls(), rows=rows, replace=replace, skip_duplicates=skip_duplicates, ignore_extra_fields=ignore_extra_fields, allow_direct_insert=allow_direct_insert)
        
        if insert_to_parts is not None:
            try:
                for part in insert_to_parts:
                    part.insert(rows=rows, **{'ignore_extra_fields': True}) if insert_to_parts_kws == {} else part.insert(rows=rows, **insert_to_parts_kws)
            except:
                traceback.print_exc()
                print('Error inserting into part table. ')

        
class PartBase(Base):
    _is_hash_name_validated = False

    def __init_subclass__(cls, **kwargs):
        cls.init_validation()

    @classmethod
    def init_validation(cls):
        """
        Validation for initialization of subclasses of abstract class PartBase. 
        """

        super().init_validation(hash_table_name=cls.hash_table_name)
    
    @classmethod
    def _hash_name_validation(cls, source='self'):
        """
        Validates hash_name and sets hash_len
        """

        part_hash_len = None
        if cls.hash_name in cls.heading.names:
            part_hash_type, part_hash_len = cls._parse_hash_name_attribute()
            
            if part_hash_type != 'varchar' or not (part_hash_len > 0 and part_hash_len <= 32):
                raise ValidationError(f'hash_name "{cls.hash_name}" must be a "varchar" type > 0 and <= 32 characters')

        master_hash_len = None
        if cls.hash_name in cls.master.heading.names:
            master_hash_type, master_hash_len = cls._parse_hash_name_attribute(source='master')
            
            if master_hash_type != 'varchar' or not (master_hash_len > 0 and master_hash_len <= 32):
                raise ValidationError(f'hash_name "{cls.hash_name}" must be a "varchar" type > 0 and <= 32 characters')

        if part_hash_len and master_hash_len:
            assert part_hash_len == master_hash_len, f'hash_name "{cls.hash_name}" varchar length mismatch. Part table length is {part_hash_len} but master length is {master_hash_len}'        
            cls._hash_len = part_hash_len
        
        elif part_hash_len:
            cls._hash_len = part_hash_len
        
        else: 
            cls._hash_len = master_hash_len

        cls._is_hash_name_validated = True

    @classmethod
    def insert_validation(cls):
        """
        Validation for insertion into subclasses of abstract class PartBase. 
        """
                    
        if cls.hash_name is not None:
            if not (cls.hash_name in cls.heading.names or cls.hash_name in cls.master.heading.names):
                raise ValidationError(f'hash_name: "{cls.hash_name}" must be present in the part table or master table heading.')
            
            # hash_name validation
            if not cls._is_hash_name_validated:
                cls._hash_name_validation()

        super().insert_validation()

    @classmethod
    def insert(cls, rows, replace=False, skip_duplicates=False, ignore_extra_fields=False, allow_direct_insert=None, reload_dependencies=False, insert_to_master=False, insert_to_master_kws={}, skip_hashing=False, constant_attrs={}, overwrite_rows=False):
        """
        Insert rows to cls.

        :params rows, replace, skip_duplicates, ignore_extra_fields, allow_direct_insert: see DataJoint insert function
        :param reload_dependencies (bool): force reload DataJoint networkx graph dependencies before insert.
        :param insert_to_master (bool): whether to insert to master table before inserting to part.
        :param insert_to_master_kws (dict): kwargs to pass to master table insert function.
        :param skip_hashing (bool): If True, hashing will be skipped if hashing is enabled. 
        :param constant_attrs (dict): Python dictionary to add to every row in rows
        :overwrite_rows (bool): Whether to overwrite key/ values in rows. If False, conflicting keys will raise a ValidationError.
        """
        assert isinstance(insert_to_master, bool), '"insert_to_master" must be a boolean.'
        
        cls.load_dependencies(force=reload_dependencies)
        
        rows = cls._prepare_insert(rows, constant_attrs=constant_attrs, overwrite_rows=overwrite_rows, skip_hashing=skip_hashing)
        
        try:
            if insert_to_master:
                cls.master.insert(rows=rows, **{'ignore_extra_fields': True, 'skip_duplicates': True}) if insert_to_master_kws == {} else cls.master.insert(rows=rows, **insert_to_master_kws)
        except:
            traceback.print_exc()
            print('Master did not insert correctly. Part insert aborted.')
            return
        
        if insert_to_master:
            try:
                super().insert(cls(), rows=rows, replace=replace, skip_duplicates=skip_duplicates, ignore_extra_fields=ignore_extra_fields, allow_direct_insert=allow_direct_insert)
            except:
                traceback.print_exc()
                print('Master inserted but part did not. Verify master inserted correctly.')
        else:
            super().insert(cls(), rows=rows, replace=replace, skip_duplicates=skip_duplicates, ignore_extra_fields=ignore_extra_fields, allow_direct_insert=allow_direct_insert)


class Lookup(MasterBase, dj.Lookup):
    pass


class Manual(MasterBase, dj.Manual):
    pass


class Computed(MasterBase, dj.Computed):
    pass


class Imported(MasterBase, dj.Imported):
    pass


class Part(PartBase, dj.Part):
    pass

# Utilities

def enable_datajoint_flags(enable_python_native_blobs=True):
    """
    Enable experimental datajoint features
    
    These flags are required by 0.12.0+ (for now).
    """
    dj.config['enable_python_native_blobs'] = enable_python_native_blobs
    dj.errors._switch_filepath_types(True)
    dj.errors._switch_adapted_types(True)


def register_externals(external_stores):
    """
    Registers external stores to DataJoint.
    """
    if 'stores' not in dj.config:
        dj.config['stores'] = external_stores
    else:
        dj.config['stores'].update(external_stores)


def make_store_dict(path):
    return {
        'protocol': 'file',
        'location': str(path),
        'stage': str(path)
    }


def _get_calling_context() -> locals:
    # get the calling namespace
    try:
        frame = inspect.currentframe().f_back
        context = frame.f_locals
    finally:
        del frame
    return context


def add_objects(objects, context=None):
    """
    Imports the adapters for a schema_name into the global namespace.
    """   
    if context is None:
        # if context is missing, use the calling namespace
        try:
            frame = inspect.currentframe().f_back
            context = frame.f_locals
        finally:
            del frame
    
    for name, obj in objects.items():
        context[name] = obj


djp_mapping = {
    'Lookup': Lookup,
    'Manual': Manual,
    'Computed': Computed,
    'Imported': Imported,
    'Part': Part
}

def add_datajoint_plus(module):
    """
    Adds DataJointPlus recursively to DataJoint tables inside the module.
    """
    try:
        for name in dir(module):
            if name in ['key_source', '_master', 'master']:
                continue
            obj = getattr(module, name)
            if inspect.isclass(obj) and issubclass(obj, UserTable) and not issubclass(obj, Base):
                bases = []
                for b in obj.__bases__:
                    if issubclass(b, UserTable):
                        b = djp_mapping[b.__name__]
                    bases.append(b)
                obj.__bases__ = tuple(bases)
                obj.parse_hash_info_from_header()
                add_datajoint_plus(obj)
    except:
        logging.warning(f'Could not add DataJointPlus to: {name}.')
        traceback.print_exc()


def reassign_master_attribute(module):
    """
    Overwrite .master attribute in DataJoint part tables to map to master class from current module. This is required if the DataJoint table is inherited.
    """
    for name in dir(module):
        # Get DataJoint tables
        if inspect.isclass(getattr(module, name)) and issubclass(getattr(module, name), dj.Table):
            obj = getattr(module, name)
            for nested in dir(obj):
                # Get Part tables
                if inspect.isclass(getattr(obj, nested)) and issubclass(getattr(obj, nested), dj.Part):
                    setattr(getattr(obj, nested), '_master', obj)


class DataJointPlusModule(dj.VirtualModule):
    """
    DataJointPlus extension of DataJoint virtual module with the added ability to instantiate from an existing module.
    """
    def __init__(self, module_name=None, schema_name=None, module=None, schema_obj_name=None, add_externals=None, add_objects=None, create_schema=False, create_tables=False, connection=None, spawn_missing_classes=True, load_dependencies=True, warn=True):
        """
        Add DataJointPlus methods to all DataJoint user tables in a DataJoint virtual module or to an existing module. 
        
        To instantiate a DataJoint Virtual Module, provide args module_name and schema_name. 
        
        To modify an existing module, provide arg module. 
        
        :param module_name (str): displayed module name (if using DataJoint Virtual module)
        :param schema_name (str): name of the database in mysql
        :param module (module): module to modify
        :param schema_obj_name (str): The name of the schema object you wish to instantiate (only needed if the module contains more than one DataJoint dj.schema object)
        :param add_externals (dict): Dictionary mapping to external files.
        :param add_objects (dict): additional objects to add to the module
        :param spawn_missing_classes (bool): Only relevant if module provided. If True, adds DataJoint tables not in module but present in mysql as classes. 
        :param load_dependencies (bool): Loads the DataJoint graph.
        :param create_schema (bool): if True, create the schema on the database server
        :param create_tables (bool): if True, module.schema can be used as the decorator for declaring new
        :param connection (dj.Connection): a dj.Connection object to pass into the schema
        :param warn (bool): if False, warnings are disabled. 
        :return: the virtual module or modified module with DataJointPlus added.
        """
        if schema_name:
            assert not module, 'Provide either schema_name or module but not both.'
            super().__init__(module_name=module_name if module_name else schema_name, schema_name=schema_name, add_objects=add_objects, create_schema=create_schema, create_tables=create_tables, connection=connection)
            
            if load_dependencies:
                self.__dict__['schema'].connection.dependencies.load()
            
        elif module:
            super(dj.VirtualModule, self).__init__(name=module.__name__)
            if module_name:
                if warn:
                    logging.warning('module_name ignored when instantiated with module.')
                
            if schema_obj_name:
                assert schema_obj_name in module.__dict__, f'schema_obj_name: {schema_obj_name} not found in module.'
                schema_obj = module.__dict__[schema_obj_name]
                assert isinstance(schema_obj, dj.Schema), f'schema object should be of type {dj.Schema} not {type(schema_obj)}.'
            else:
                schemas = {k: {'obj': v, 'database': v.database} for k, v in module.__dict__.items() if isinstance(v, dj.Schema)}
                assert len(schemas.keys())==1, f"Found multiple schema objects with names {list(schemas.keys())}, mapping to respective databases {[v['database'] for v in schemas.values()]}. Specify the name of the schema object to instantiate with arg schema_obj_name." 
                schema_obj = list(schemas.values())[0]['obj']
            
            self.__dict__.update(module.__dict__)
            self.__dict__['schema'] = schema_obj
            
            if spawn_missing_classes:
                schema_obj.spawn_missing_classes(context=self.__dict__)
                
            if load_dependencies:
                schema_obj.connection.dependencies.load()
                
            if add_objects:
                self.__dict__.update(add_objects)
        
        else:
            raise ValueError('Provide schema_name or module.')      
        
        if add_externals:
            register_externals(add_externals)
            
        add_datajoint_plus(self)
    
create_djp_module = DataJointPlusModule