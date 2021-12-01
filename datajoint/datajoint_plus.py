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
import warnings


__version__ = "0.0.8"


class classproperty:
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


class ValidationError(dj.DataJointError):
    pass


class OverwriteError(dj.DataJointError):
    pass

_vm_modification_err = 'Table modification not allowed with virtual modules. '

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
    is_insert_validated = False,
    enable_hashing = False,
    hash_name = None,
    hashed_attrs = None,
    hash_group = False,
    add_hash_name_to_header = True,
    add_hashed_attrs_to_header = True,
    _hash_len = None

    @classmethod
    def init_validation(cls):
        """
        Validation for initialization of subclasses of abstract class Base. 
        """
        for attr in ['enable_hashing', 'hash_group', 'add_hash_name_to_header', 'add_hashed_attrs_to_header']:
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
        
        if cls.hash_name is not None or cls.hashed_attrs is not None:
            if cls.add_hash_name_to_header or cls.add_hashed_attrs_to_header:
                cls._add_hash_info_to_header(add_hash_name=cls.add_hash_name_to_header if cls.hash_name is not None else False, add_hashed_attrs=cls.add_hashed_attrs_to_header if cls.hashed_attrs is not None else False)

    @classmethod
    def insert_validation(cls):
        """
        Validation for insertion to DataJoint tables that are subclasses of abstract class Base. 
        """
        assert cls.__module__ != 'datajoint.user_tables', _vm_modification_err

        # ensure hash_name and hashed_attrs are disjoint
        if cls.hash_name is not None and cls.hashed_attrs is not None:
            if not set((cls.hash_name,)).isdisjoint(cls.hashed_attrs):
                raise NotImplementedError(f'attributes in "hash_name" and "hashed_attrs" must be disjoint.')

        cls.is_insert_validated = True

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
          
    @staticmethod
    def _hash_name_type_validation(hash_name, hash_name_type):
        """
        Validates hash_name type and returns hash_len
        """
        hash_name_type_error_msg = f'hash_name "{hash_name}" must be a "varchar" type > 0 and <= 32 characters'
        
        hash_name_type_parsed = re.findall('\w+', hash_name_type)
        if 'varchar' not in hash_name_type_parsed:
            raise ValidationError(hash_name_type_error_msg)

        # hash_name varchar length validation
        hash_len = int(hash_name_type_parsed[1])
        assert (hash_len > 0 and hash_len <= 32), hash_name_type_error_msg
        
        return hash_len

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
    def _add_hash_info_to_header(cls, add_hash_name=True, add_hashed_attrs=True):
        """
        Modifies definition header to include hash_name and hashed_attrs with a parseable syntax. 

        :param add_hash_name (bool): Whether to add hash_name to header
        :param add_hashed_attrs (bool): Whether to add hashed_attrs to header
        """
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
    def add_hash_to_rows(cls, rows, hash_table_name=False, overwrite_rows=False):
        """
        Adds hash to rows.

        :param rows (pd.DataFrame, QueryExpression, list, tuple): rows to pass to DataJoint `insert`.
        :param hash_table_name (bool): Whether to include table_name in rows for hashing
        :overwrite_rows (bool): Whether to overwrite key/ values in rows. If False, conflicting keys will raise a ValidationError. 

        :returns: modified rows
        """      
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
                rows[cls.hash_name] = generate_hash(rows_to_hash, add_constant_columns=table_name)[:cls._hash_len]

            else:
                rows[cls.hash_name] = [generate_hash([row], add_constant_columns=table_name)[:cls._hash_len] for row in rows_to_hash.to_dict(orient='records')]
                
        return rows

    @classmethod
    def _prepare_insert(cls, rows, constant_attrs, overwrite_rows=False, skip_hashing=False):
        """
        Prepares rows for insert by checking if table has been validated for insert, adds constant_attrs and performs hashing. 
        """
        
        if not cls.is_insert_validated:
            cls.insert_validation()
        
        if constant_attrs != {}:
            rows = cls.add_constant_attrs_to_rows(rows, constant_attrs, overwrite_rows)

        if cls.enable_hashing and not skip_hashing:
            try:
                hash_table_name = True if (issubclass(cls, MasterBase) and cls.hash_table_name) or (issubclass(cls, dj.Part) and cls.master.hash_part_table_names) else False
                rows = cls.add_hash_to_rows(rows, hash_table_name=hash_table_name, overwrite_rows=overwrite_rows)

            except OverwriteError as err:
                new = err.args[0]
                new += ' Or, to skip the hashing step, set skip_hashing=True.'
                raise OverwriteError(new) from None

        return rows


class MasterBase(Base):
    hash_table_name = False
    hash_part_table_names = False

    def __init_subclass__(cls, **kwargs):
        cls.init_validation()

    @classmethod
    def init_validation(cls):
        """
        Validation for initialization of subclasses of abstract class MasterBase. 
        """
        for attr in ['hash_table_name', 'hash_part_table_names']:
            assert isinstance(getattr(cls, attr), bool), f'"{attr}" must be a boolean.'

        super().init_validation()

    @classmethod
    def insert_validation(cls):
        """
        Validation for insertion into subclasses of abstract class MasterBase. 
        """
        if cls.hash_name is not None:
            if cls.hash_name not in cls.heading.names:
                raise ValidationError(f'Attribute "{cls.hash_name}" in property "hash_name" must be present in table heading.')

            # hash_name type validation
            cls._hash_len = cls._hash_name_type_validation(cls.hash_name, cls.heading.attributes[cls.hash_name].type)
        
        super().insert_validation()

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
                warnings.warn('Part table defined in class definition not found in DataJoint graph. Reload dependencies.')

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
    def union_parts(cls, part_restr={}, include_parts=None, exclude_parts=None, parts_kws={}):
        """
        Returns union of part table primary keys after optional restriction. Requires all part tables in union to have identical primary keys. 

        :params: see `restrict_parts`.

        :returns: numpy array object
        """  
        return np.sum([p.proj() for p in cls.restrict_parts(include_parts=include_parts, exclude_parts=exclude_parts, part_restr=part_restr, parts_kws=parts_kws)])

#     @classmethod
#     def keys_not_in_parts(cls, part_restr={}, include_parts=None, exclude_parts=None, master_restr={}, parts_kws={}):
#         return (cls & master_restr) - cls.union_parts(include_parts=include_parts, exclude_parts=exclude_parts, part_restr=part_restr, parts_kws=parts_kws)

    @classmethod
    def join_parts(cls, part_restr={}, include_parts=None, exclude_parts=None, join_method=None, join_with_master=False, parts_kws={}):
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
        parts = cls.restrict_parts(include_parts=include_parts, exclude_parts=exclude_parts, part_restr=part_restr, parts_kws=parts_kws)
        
        if join_with_master:
            parts = [dj.FreeTable(cls.connection, cls.full_table_name)] + parts

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
    def restrict_one_part(cls, part_restr={}, include_parts=None, exclude_parts=None, parts_kws={}):
        """
        Calls `restrict_parts` with filter_out_len_zero=True. If not exactly one part table is returned, then a ValidationError will be raised.

        WARNING: If the attributes in part and part_restr are mutually exclusive, then len(part & part_restr) > 0. 
        This means that if a part table and part_restr don't share any column names, then the part table will not be filtered out from `restrict_parts` even if the part_restr has no matching entries in that part table.

        :params: see `restrict_parts`.

        :returns: part table after restriction.
        """
        parts = cls.restrict_parts(part_restr=part_restr, include_parts=include_parts, exclude_parts=exclude_parts, filter_out_len_zero=True, parts_kws=parts_kws)

        if len(parts) > 1:
            raise ValidationError('part_restr can restrict multiple part tables.')
        
        elif len(parts) < 1:
            raise ValidationError('part_restr can not restrict any part tables.')
        
        else:
            return parts[0]

    r1p = restrict_one_part # alias for restrict_one_part

    @classmethod
    def part_table_names_with_hash(cls, hash, hash_name=None, include_parts=None, exclude_parts=None, parts_kws={}):
        """
        Calls `restrict_parts_with_hash` with filter_out_len_zero=True.

        :params: see `restrict_parts_with_hash`

        :returns: list of part table names that contain hash.
        """
        parts = cls.restrict_parts_with_hash(hash=hash, hash_name=hash_name, include_parts=include_parts, exclude_parts=exclude_parts, filter_out_len_zero=True, parts_kws=parts_kws)
        return [format_table_name(r.table_name, part=True) for r in parts]

    @classmethod
    def restrict_one_part_with_hash(cls, hash, hash_name=None, include_parts=None, exclude_parts=None, parts_kws={}):
        """
        Calls `restrict_parts_with_hash` with filter_out_len_zero=True. If not exactly one part table is returned, then a ValidationError will be raised.

        :params: see `restrict_parts_with_hash`

        :returns: part table after restriction
        """
        parts = cls.restrict_parts_with_hash(hash=hash, hash_name=hash_name, include_parts=include_parts, exclude_parts=exclude_parts, filter_out_len_zero=True, parts_kws=parts_kws)

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
        if not cls.is_insert_validated:
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
    def __init_subclass__(cls, **kwargs):
        cls.init_validation()

    @classmethod
    def init_validation(cls):
        """
        Validation for initialization of subclasses of abstract class PartBase. 
        """
        if hasattr(cls, 'hash_table_name'):
            raise ValidationError(f'Part tables cannot contain "hash_table_name" property. To hash the table name of part tables, set hash_part_table_names=True in master table.')

        super().init_validation()

    @classmethod
    def insert_validation(cls):
        """
        Validation for insertion into subclasses of abstract class PartBase. 
        """
        
        for set_default in ['hash_part_table_names']:
            if not hasattr(cls.master, set_default):
                setattr(cls.master, set_default, False)
            else:
                if not isinstance(getattr(cls.master, set_default), bool):
                    raise NotImplementedError(f'"{set_default}" must be boolean.') 
                    
        if cls.hash_name is not None:
            if not (cls.hash_name in cls.heading.names or cls.hash_name in cls.master.heading.names):
                raise ValidationError(f'Attribute "{cls.hash_name}" in "hash_name" must be present in the part table or master table heading.')

            part_hash_len = None
            if cls.hash_name in cls.heading.names:
                part_hash_len = cls._hash_name_type_validation(cls.hash_name, cls.heading.attributes[cls.hash_name].type)

            master_hash_len = None
            if cls.hash_name in cls.master.heading.names:
                master_hash_len = cls._hash_name_type_validation(cls.hash_name, cls.master.heading.attributes[cls.hash_name].type)

            if (part_hash_len is not None) and (master_hash_len is not None):
                assert part_hash_len == master_hash_len, f'hash_name "{cls.hash_name}" varchar length mismatch. Part table length is {part_hash_len} but master length is {master_hash_len}'
                cls._hash_len = part_hash_len

            elif part_hash_len is not None:
                cls._hash_len = part_hash_len

            else:
                cls._hash_len = master_hash_len

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

# VIRTUAL MODULES

class VirtualModule:   
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
                if result[0] == 'hashed_attrs':
                    cls.hashed_attrs = result[1:]

    @classmethod
    def insert(cls, *args, **kwargs):
        raise AttributeError(_vm_modification_err)
    
    @classmethod
    def _update(cls, *args, **kwargs):
        raise AttributeError(_vm_modification_err)
    
    @classmethod
    def delete(cls, *args, **kwargs):
        raise AttributeError(_vm_modification_err)
    
    @classmethod
    def delete_quick(cls, *args, **kwargs):
        raise AttributeError(_vm_modification_err)
    
    @classmethod
    def drop(cls, *args, **kwargs):
        raise AttributeError(_vm_modification_err)
    
    @classmethod
    def drop_quick(cls, *args, **kwargs):
        raise AttributeError(_vm_modification_err)
    

class VirtualLookup(VirtualModule, Lookup):
    pass


class VirtualManual(VirtualModule, Manual):
    pass


class VirtualComputed(VirtualModule, Computed):
    pass


class VirtualImported(VirtualModule, Imported):
    pass


class VirtualPart(VirtualModule, Part):
    pass