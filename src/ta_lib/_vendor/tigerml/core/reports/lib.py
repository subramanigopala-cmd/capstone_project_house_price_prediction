def create_report(
    contents,
    name="",
    path="",
    format=".html",
    split_sheets=True,
    tiger_template=False,
    **kwargs,
):
    """Create a TigerML style report in html, excel or ppt format.

    This function generates a report file of specified format in the given folder
    path. Use kwargs to pass any other arguments to be passed into respective
    report engines for html, excel or ppt.

    Parameters
    ----------
    contents : dict
        A dict or nested dict of tables (pd.DataFrame) or plots to be added to report.
        The dictionary keys should be specified in such a way that they are used as
        titles of sections/sub-sections of the report.

    name : str
        A string specifying the name of the report (without the extension).

    path : str
        Path to the folder where report needs to be saved (without the filename
        or extension).

    format : str; default is ".html"
        Format of the report. It can take values ".html", ".xlsx" or ".pptx".
        Default format is ".html". For other formats, additional dependencies
        may be required.

    split_sheets : bool; default is True
        Specify whether to split the report in multiple sheets or sections based
        on the nesting defined in the contents dict.

    tiger_template : bool; default is False
        Specify whether a tiger specific ppt template is available (required only
        for ppt report). If True, it looks for a template file "tiger_template.pptx".

    **kwargs : dict, optional
        Additional arguments, including:

        - `save` (bool, optional, default True):
            - If `True`, the report will be saved to the specified path.
            - If `False`, for HTML reports, the HTML content will be returned as a string, otherwise, no report is saved.
            - For Excel/PowerPoint formats, controls only saving behavior.

    Returns
    -------
    result : str or None
        - For HTML reports: Returns the HTML content as a string if `save=False`, otherwise `None`.
        - For Excel/PowerPoint reports: Always returns `None` after saving the report.

    Notes
    -----
    - For HTML reports, `save=False` prevents saving and returns the HTML content.
    - For Excel/PowerPoint reports, `save=False` prevents saving, and `None` is returned.
    """
    if format == ".xlsx":
        from .excel import create_excel_report

        create_excel_report(
            contents, name=name, path=path, split_sheets=split_sheets, **kwargs
        )
    elif format == ".pptx":
        from .ppt.lib import create_ppt_report

        create_ppt_report(contents, name=name, path=path, tiger_template=tiger_template)
    if format == ".html":
        from .html import create_html_report

        if "save" in kwargs.keys() and kwargs["save"] is False:
            return create_html_report(
                contents, name=name, path=path, split_sheets=split_sheets, **kwargs
            ).get_html_content()
        else:
            return create_html_report(
                contents, name=name, path=path, split_sheets=split_sheets, **kwargs
            )
