class Reference(object):
    """ A container for a name and subindexing the object that name refers to """
    def __init__(self, name, slices=None):
        self.name = name
        if slices:
            self.slices = slices
        else:
            self.slices = ()

    def __getitem__(self, s):
        if isinstance(s, tuple):
            sub_ref = Reference(self.name, self.slices + s)
        else:
            sub_ref = Reference(self.name, self.slices + (s,))
        return sub_ref

    def __repr__(self):
        return self.name + ''.join([
            '[%s]' % i for i in self.slices
        ])

    def retrieve_index(self, item):
        for s in self.slices:
            item = item[s]
        return item

    def set_index(self, item, value, copy=False):
        if copy:
            item = item.copy()
        # in order to ensure we change the data structure and not just the pointer
        # we follow the slice tree to the 2nd to last reference,
        # then use the last reference as a way of resetting the pointer
        for i in range(len(self.slices)-1):
            item = item[s]
        item[self.slices[-1]] = value
        return item

    def is_base(self):
        return len(self.slices) == 0

    def __hash__(self):
        return hash(self.__class__) + hash(self.name) + hash(self.slices)

    def __eq__(self, other):
        return (self.name == other.name) and (self.slices == other.slices)