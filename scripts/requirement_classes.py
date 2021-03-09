class Requirement:
    def __init__(self,isFunctional=None):
        self._id = None
        self._isFunctional = isFunctional
    
    def __generate_id(self):
        pass


class NonfunctionalRequirement(Requirement):
    def __init__(self,text, attribute=None, product=None, text=None, criticality=0):
        super().__init__(self,isFunctional = False)

        self._attibute = attribute
        self._product = product
        self._primarySubject = None
        self._subject = None
        self._criticality = criticality
        self._adjuncts = []
        self._text = text

        self.__generate_id()
    
    def __extract_parts_of_speech(self):
        pass

    def __identify_adjuncts(self):
        pass


class FunctionalRequirement(Requirement):
    def __init__(self,text):
        super().__init__(self,isFunctional = True)

        self._behavior = None
        self._product = None
        self._criticality = 0
        self._text = text

        self.__generate_id()
    
    def __determine_behavior(self):
        pass


class Behavior:
    def __init__(self):
        self._primarySubject = None
        self._subject = None
        self._primaryObject = None
        self._object = None
        self._verb = None
        self._adjuncts = []
    
    def __extract_parts_of_speech(self):
        pass

    def __identify_adjuncts(self):
        pass


class Adjunct:
    def __init__(self,value):
        self._type = None
        self._value = value
    
    def __determine_type(self):
        pass


class QuantityAttribute:
    def __init__(self,text=None):
        self._text = text
        self._unit = None
        self._lowerValue = None
        self._upperValue = None


class QualityAttribute:
    def __init__(self,value):
        self._value = value


class Part:
    def __init__(self,name):
        self._id = None
        self._name = name
    
    def __generate_id(self):
        pass


class Product(Part):
    def __init__(self, name):
        super().__init__(self,name)

        self._parts = []

        self.__generate_id()