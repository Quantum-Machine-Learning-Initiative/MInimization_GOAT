Fresco, a web micro-framework for Python
========================================

The fresco web framework is:

- Fast, simple and powerful, with sophisticated URL routing, request and response objects.
- Lightweight and open to integration: you pick the templating and database libraries you want.
- WSGI compliant: easy integration with your choice of web server, apps and middleware.

A minimal fresco web framework application:

.. code-block:: python

    from fresco import FrescoApp, GET, Response

    def helloworld():
        return Response("<h1>Hello World!</h1>")

    app = FrescoApp()
    app.route('/', GET, helloworld)


Read the
`fresco web framework documentation
<http://www.ollycope.com/software/fresco/latest/>`_ for
more about the framework, or
visit the `bitbucket repo <https://bitbucket.com/ollyc/fresco/>`_.


