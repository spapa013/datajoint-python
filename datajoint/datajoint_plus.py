from os import confstr_names, error
import datajoint as dj
from datajoint.expression import QueryExpression
import inspect
import collections
import pandas as pd
import hashlib
import json
import re
from enum import Enum
import numpy as np
from collections import Counter
from IPython.display import display, clear_output
from ipywidgets.widgets import Output, HBox, Label
import traceback
import copy
import warnings


__version__ = "0.0.2"

class classproperty:
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


class ValidationError(dj.DataJointError):
    pass


class OverwriteError(dj.DataJointError):
    pass


def generate_hash(rows, add_dict_to_all_rows:dict=None):
    df = pd.DataFrame(rows)
    if add_dict_to_all_rows is not None:
        assert isinstance(add_dict_to_all_rows, dict)
        for k, v in add_dict_to_all_rows.items():
            df[k] = v
    df = df.sort_index(axis=1).sort_values(by=[*df.columns])
    encoded = json.dumps(df.to_dict(orient='records')).encode()
    dhash = hashlib.md5()
    dhash.update(encoded)
    return dhash.hexdigest()


def _validate_rows_for_hashing(rows):
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
    _validate_rows_for_hashing(rows)
    return generate_hash(rows, **kwargs)
    

def split_full_table_name(full_table_name):
    """
    param: full_table_name
    returns: database, table_name
    """
    return tuple(s.strip('`') for s in full_table_name.split('.'))


def format_table_name(table_name, snake_case=False, part=False):
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
    is_insert_validated = False
    enable_hashing = False
    hash_name = None
    hashed_attrs = None
    hash_group = False
    add_hash_info_to_header = True
    _hash_len = None

    @classmethod
    def init_validation(cls):
        assert hasattr(cls, 'enable_hashing') and isinstance(cls.enable_hashing, bool), 'Subclasses of Base must implement boolean property "enable_hashing".'
                
        if cls.enable_hashing:
            for required in ['hash_name', 'hashed_attrs', 'hash_group', 'add_hash_info_to_header']:
                if not hasattr(cls, required) or getattr(cls, required) is None:
                    raise NotImplementedError(f'Hashing requires class to implement the property "{required}".')
            
            # ensure one attribute for "hash_name"
            if type(cls.hash_name) == list or type(cls.hash_name) == tuple:
                if len(cls.hash_name) > 1:
                    raise NotImplementedError(f'Only one attribute allowed in "hash_name".')
                else:
                    cls.hash_name = cls.hash_name[0]

            # ensure "hashed_attrs" wrapped in list or tuple
            if not isinstance(cls.hashed_attrs, list) and not isinstance(cls.hashed_attrs, tuple):
                cls.hashed_attrs = [cls.hashed_attrs]
            else:
                cls.hashed_attrs = cls.hashed_attrs

            # ensure hash_name and hashed_attrs are disjoint
            if not set((cls.hash_name,)).isdisjoint(cls.hashed_attrs):
                raise NotImplementedError(f'attributes in "hash_name" and "hashed_attrs" must be disjoint.')

            # ensure hash_group is bool
            if not isinstance(cls.hash_group, bool):
                raise NotImplementedError(f'property "hash_group" must be boolean.')

            if cls.add_hash_info_to_header:
                cls._add_hash_info_to_header()
  
    @classmethod
    def insert_validation(cls):
        cls.is_insert_validated = True

    @classmethod
    def load_dependencies(cls, force=False):
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
        assert isinstance(constant_attrs, dict), 'arg "constant_attrs" must be a dict type.'
        
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
        return cls.proj(..., **{a: '""' for a in cls.heading.names if a not in args}).proj(*[a for a in cls.heading.names if a in args])
    
    @classmethod
    def exclude_attrs(cls, *args):
        return cls.proj(..., **{a: '""' for a in cls.heading.names if a in args}).proj(*[a for a in cls.heading.names if a not in args])
          
    @staticmethod
    def _hash_name_type_validation(hash_name, hash_name_type_parsed):
        hash_name_type_error_msg = f'hash_name "{hash_name}" must be a "varchar" type > 0 and <= 32 characters'

        if 'varchar' not in hash_name_type_parsed:
            raise ValidationError(hash_name_type_error_msg)

        # hash_name varchar length validation
        hash_len = int(hash_name_type_parsed[1])
        assert (hash_len > 0 and hash_len <= 32), hash_name_type_error_msg
        
        return hash_len

    @classmethod
    def restrict_with_hash(cls, hash, hash_name=None):
        if hash_name is None and hasattr(cls, 'hash_name'):
            hash_name = cls.hash_name

        if hash_name is None:
            raise ValidationError('Table does not have "hash_name" defined, provide it to restrict with hash.')
            
        return cls & {cls.hash_name: hash}

    @classmethod
    def _add_hash_info_to_header(cls):
        inds, contents, _ = parse_definition(cls.definition)
        headers = contents['headers']

        if len(headers) >= 1:
            header = headers[0]

        else:
            # create header
            header = """#"""

        # append hash info to header 
        header += f" | hash_name = {cls.hash_name}; hashed_attrs = "
        for i, h in enumerate(cls.hashed_attrs):
            header += f"{h}, " if i+1 < len(cls.hashed_attrs) else f"{h}"
        
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
        if not cls.is_insert_validated:
            cls.insert_validation()
        
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
                rows[cls.hash_name] = generate_hash(rows_to_hash, add_dict_to_all_rows=table_name)[:cls._hash_len]

            else:
                rows[cls.hash_name] = [generate_hash([row], add_dict_to_all_rows=table_name)[:cls._hash_len] for row in rows_to_hash.to_dict(orient='records')]
                
        return rows

    @classmethod
    def _prepare_insert(cls, rows, constant_attrs, overwrite_rows=False, skip_hashing=False):
        
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
        if cls.enable_hashing:
            for required in ['hash_table_name', 'hash_part_table_names']:
                if not hasattr(cls, required) or getattr(cls, required) is None:
                    raise NotImplementedError(f'Hashing requires class to implement the property "{required}".')

        super().init_validation()

    @classmethod
    def insert_validation(cls):
        if cls.enable_hashing:
            if cls.hash_name not in cls.heading.names:
                raise ValidationError(f'Attribute "{cls.hash_name}" in property "hash_name" must be present in table heading.')

            # hash_name type validation
            cls._hash_len = cls._hash_name_type_validation(cls.hash_name, re.findall('\w+', cls.heading.attributes[cls.hash_name].type))
        
        super().insert_validation()

        
    @classmethod
    def parts(cls, as_objects=False, as_cls=False, reload_dependencies=False):
        cls.load_dependencies(force=reload_dependencies)

        cls_parts = [getattr(cls, d) for d in dir(cls) if inspect.isclass(getattr(cls, d)) and issubclass(getattr(cls, d), dj.Part)]
        for cls_part in [p.full_table_name for p in cls_parts]:
            if cls_part not in super().parts(cls):
                warnings.warn('Part table defined in class definition not found in DataJoint graph. Consider running again with reload_dependencies=True.')

        if not as_cls:
            return super().parts(cls, as_objects=as_objects)
        else:
            return cls_parts

        
    @classmethod
    def number_of_parts(cls, parts_kws={}):
        return len(cls.parts(**parts_kws))

    
    @classmethod
    def has_parts(cls, parts_kws={}):
        return cls.number_of_parts(parts_kws) > 0

    
    @classmethod
    def _format_parts(cls, parts):
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
    def restrict_parts(cls, part_restr={}, include_parts=None, exclude_parts=None, parts_kws={}):
        assert cls.has_parts(parts_kws), 'No part tables found.'
        parts_kws = {k:v for k,v in parts_kws.items() if k not in ['reload_dependencies']}

        if include_parts is None:
            parts = cls.parts(**parts_kws) if parts_kws!={} else cls.parts(as_cls=True)
        
        else:
            parts = cls._format_parts(include_parts)
        
        if exclude_parts is not None:
            parts = [p for p in parts if p.full_table_name not in [e.full_table_name for e in cls._format_parts(exclude_parts)]]

        return [p & part_restr for p in parts]

    @classmethod
    def union_parts(cls, part_restr={}, include_parts=None, exclude_parts=None, parts_kws={}):        
        return np.sum([p.proj() for p in cls.restrict_parts(include_parts=include_parts, exclude_parts=exclude_parts, part_restr=part_restr, parts_kws=parts_kws)])

#     @classmethod
#     def keys_not_in_parts(cls, part_restr={}, include_parts=None, exclude_parts=None, master_restr={}, parts_kws={}):
#         return (cls & master_restr) - cls.union_parts(include_parts=include_parts, exclude_parts=exclude_parts, part_restr=part_restr, parts_kws=parts_kws)

    @classmethod
    def join_parts(cls, part_restr={}, include_parts=None, exclude_parts=None, join_method=None, join_with_master=False, parts_kws={}):
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
                for i, f in enumerate(JoinMethod):
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
            for i, f in enumerate(JoinMethod):
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
    def parts_with_hash(cls, hash, hash_name=None, include_parts=None, exclude_parts=None, parts_kws={}):
        return [format_table_name(r.table_name, part=True) for r in cls.restrict_parts_with_hash(hash, hash_name, include_parts, exclude_parts, parts_kws)]

    @classmethod
    def restrict_one_part_with_hash(cls, hash, hash_name=None, include_parts=None, exclude_parts=None, parts_kws={}):
        restrs = cls.restrict_parts_with_hash(hash, hash_name, include_parts, exclude_parts, parts_kws)
        
        if len(restrs) > 1:
            raise ValidationError('Hash found in multiple part tables.')
        
        elif len(restrs) < 1:
            raise ValidationError('Hash not found in any part tables.')
        
        return restrs[0]
    
    r1pwh = restrict_one_part_with_hash # alias for restrict_one_part_with_hash

    @classmethod
    def restrict_parts_with_hash(cls, hash, hash_name=None, include_parts=None, exclude_parts=None, parts_kws={}):       
        if hash_name is None and hasattr(cls, 'hash_name'):
            hash_name = cls.hash_name

        if hash_name is None:
            raise ValidationError('Table does not have "hash_name" defined, provide it to restrict with hash.')
        
        parts = cls.restrict_parts(part_restr={hash_name: hash}, include_parts=include_parts, exclude_parts=exclude_parts, parts_kws=parts_kws)
        return [p for p in parts if hash_name in p.heading.names and len(p)>0]
        

    @classmethod
    def insert(cls, rows, replace=False, skip_duplicates=False, ignore_extra_fields=False, allow_direct_insert=None, reload_dependencies=False, insert_to_parts=None, insert_to_parts_kws={}, skip_hashing=False, constant_attrs={}, overwrite_rows=False):
        if not cls.is_insert_validated:
            cls.insert_validation()
        
        if insert_to_parts is not None:
            assert cls.has_parts(reload_dependencies=reload_dependencies), 'No part tables found.'
            insert_to_parts = cls._format_parts(insert_to_parts)

        rows = cls._prepare_insert(rows, constant_attrs=constant_attrs, overwrite_rows=overwrite_rows, skip_hashing=skip_hashing)

        cls._insert(rows=rows, replace=replace, skip_duplicates=skip_duplicates, ignore_extra_fields=ignore_extra_fields, allow_direct_insert=allow_direct_insert, insert_to_parts=insert_to_parts, insert_to_parts_kws=insert_to_parts_kws)
    
    @classmethod
    def _insert(cls, rows, replace=False, skip_duplicates=False, ignore_extra_fields=False, allow_direct_insert=False, insert_to_parts=None, insert_to_parts_kws={}):
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
        if cls.enable_hashing:
            for required in ['hash_name', 'hashed_attrs', 'hash_group', 'add_hash_info_to_header']:
                if not hasattr(cls, required) or getattr(cls, required) is None:
                    raise NotImplementedError(f'Hashing requires class to implement the property "{required}".')

            if hasattr(cls, 'hash_table_name'):
                raise ValidationError(f'Part tables cannot contain "hash_table_name" property. To hash the table name of part tables, set hash_part_table_names=True in master table.')

        super().init_validation()

    @classmethod
    def insert_validation(cls):
        if cls.enable_hashing:
            if not (cls.hash_name in cls.heading.names or cls.hash_name in cls.master.heading.names):
                raise ValidationError(f'Attribute "{cls.hash_name}" in property "hash_name" must be present in the part table or master table heading.')

            for set_default in ['hash_part_table_names']:
                if not hasattr(cls.master, set_default):
                    setattr(cls.master, set_default, False) 

            part_hash_len = None
            if cls.hash_name in cls.heading.names:
                part_hash_len = cls._hash_name_type_validation(cls.hash_name, re.findall('\w+', cls.heading.attributes[cls.hash_name].type))

            master_hash_len = None
            if cls.hash_name in cls.master.heading.names:
                master_hash_len = cls._hash_name_type_validation(cls.hash_name, re.findall('\w+', cls.master.heading.attributes[cls.hash_name].type))

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
        assert isinstance(insert_to_master, bool), 'insert_to_master must be a boolean.'
        
        cls.load_dependencies(force=reload_dependencies)
        
        rows = cls._prepare_insert(rows, constant_attrs=constant_attrs, overwrite_rows=overwrite_rows, skip_hashing=skip_hashing)
        
        cls._insert(rows=rows, replace=replace, skip_duplicates=skip_duplicates, ignore_extra_fields=ignore_extra_fields, allow_direct_insert=allow_direct_insert, insert_to_master=insert_to_master, insert_to_master_kws=insert_to_master_kws)

    @classmethod
    def _insert(cls, rows, replace=False, skip_duplicates=False, ignore_extra_fields=False, allow_direct_insert=False, insert_to_master=False, insert_to_master_kws={}):
        restrict_parts
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