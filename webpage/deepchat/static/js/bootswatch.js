/* Macro for using Bootswatch API (theme-changing). */

/** Load JSON-encoded data from api server (using ajax GET HTTP request). */
$.getJSON("https://bootswatch.com/api/3.json", function (data) {

    // Get array of themes. Each entry is an object with properties:
    // - name, description, preview, thumbnail, css, cssMin, cssCdn, less, lessVariables.
    var themes = data.themes;

    // Grab select object(s).
    var select = $("#theme-options");
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
        select.append($("<option/>")
                .attr('class', 'dropdown-item')
                .val(index)          // value=index
                .text(value.name)); // displayed text = theme name
    });

    // Make a link to the theme upon selction.
    select.on('change', function(){
        var theme = themes[$(this).val()];
        $("link#theme-link").attr("href", theme.css);
    });

    // Set default value.
    // TODO: make persist across pages.
    select.val('4');


}, "json").fail(function(){
    $(".alert").toggleClass("alert-info alert-danger");
    $(".alert h4").text("Failure!");
});
