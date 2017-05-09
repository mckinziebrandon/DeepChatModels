/* Assign certain elements to bootstrap classes by default, so
 * less typing for me.
 */
$(document).ready(function() {

    // Pretty tables.
    $('table').addClass('table table-striped table-hover');

    // Lists.
    $('ul.custom').addClass('list-group');
    $('ul.custom li').addClass('list-group-item');

    // Muh tooltips.
    let tooltip = $('.tooltip-custom');
    // Only auto-show (for 3 sec) if on homepage.
    if (tooltip.attr('page') == 'index') {
        tooltip.tooltip('show');
        setTimeout(function() {
            tooltip.tooltip('hide');
        }, 3000);
    }



});
