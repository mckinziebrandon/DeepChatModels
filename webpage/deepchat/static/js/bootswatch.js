/* Macro for using Bootswatch API (theme-changing). */

/** Load JSON-encoded data from api server (using ajax GET HTTP request). */
$.getJSON("https://bootswatch.com/api/3.json", function (data) {

    // Get array of themes. Each entry is an object with properties:
    // - name, description, preview, thumbnail, css, cssMin, cssCdn, less, lessVariables.
    var themes = data.themes;

    // Grab select object(s).
    var themeList = $("ul#theme-options");
    //select.hide();

    // .show(): simplest way to display an element.
    // Doesn't seem to do anything in this case?

    // .toggleClass(): if the element has these classes, they are removed.
    // For the class names that it doesn't have, they are added.
    //$(".alert").toggleClass("alert-info alert-success");

    // prepend the word success.
    //$(".alert h4").text("Success!");

    // Fill select with theme options.
    // Array.prototype.forEach documentation:
    // - arr.forEach(function callback(currentValue, index, array) {
    //      your iterator }[, thisArg]);
    themes.forEach(function(value, index){
        /*
        themeList.append($("<option/>")
                .attr('class', 'dropdown-item')
                .val(index)          // value=index
                .text(value.name)); // displayed text = theme name
                */
        themeList.append($('<li/>')
                .attr('class', 'thing')
                .val(index)
                .html('<a href="#">'+value.name+'</a>'));
    });

    themeList.on('click', '.thing', function(e) {
        var theme = themes[$(this).val()];
        $("link#theme-link").attr("href", theme.css);
        $('a#theme-header').text('Theme: ' + theme.name);
    });

    // Set default value.
    // TODO: make persist across pages.
    var theme = themes[4];
    $("link#theme-link").attr("href", theme.css);
    $('a#theme-header').text('Theme: ' + theme.name);

}, "json").fail(function(){
    $(".alert").toggleClass("alert-info alert-danger");
    $(".alert h4").text("Failure!");
});
