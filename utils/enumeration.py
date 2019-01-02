class Enumeration:
    """
    Helps to make (lightweight) work with custom enum classes easier.
    This class is intended to be subclassed by custom enum classes. Example: class WorkDays(Enumeration)
    Enum values in custom subclass should be defined as class variables, e.g.:
    var1 = "Monday", var2 = 500, var3 = 0.1
    """
    _enum_values = []

    @classmethod
    def enum_values(cls, refresh=False):
        """
        Get list of all values in the enum.
        Note: Enum values are loaded only once by default - at first use (enum is not supposed to change
        dynamically). To refresh/load up-to-date values when calling this, use 'refresh' parameter.
        """
        # load enum values once and save them for future usage (unless forced refresh)
        if refresh or not cls._enum_values:  # if forced refresh or enum values not saved yet, load them
            values = []
            for name, value in cls.__dict__.items():
                if not name.startswith('__'):
                    values.append(value)

            cls._enum_values = values

        return cls._enum_values

    @classmethod
    def contains(cls, item, refresh=False, ignore_case=False):
        """Check if given item (value) is contained in enum."""
        if (ignore_case is True) and isinstance(item, str):
            item = item.lower()
            for value in cls.enum_values(refresh=refresh):
                if isinstance(value, str) and (item == value.lower()):
                    # item matched a value
                    return True

            # item did not match any value
            return False
        else:
            return True if item in cls.enum_values(refresh=refresh) else False


""" USAGE EXAMPLE """
if __name__ == '__main__':
    # create custom enum class
    class WorkDay(Enumeration):
        mon = 'Monday'
        tue = 'Tuesday'
        wed = 'Wednesday'
        thu = 'Thursday'
        fri = 'Friday'

    # print all items in the enum
    print("Work days are:")
    print(WorkDay.enum_values())
