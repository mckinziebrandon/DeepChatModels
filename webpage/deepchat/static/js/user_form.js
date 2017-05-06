$(document).ready(function() {

    $('#user-form-submit').on('click', function(e) {
        var userName = $('#user-name');
        if (!userName.val()) {  // don't do anything on empty input
            return;
        }
        // Submit a POST request to collect user input.
        $.post('/user/', {"name": userName.val()}, function(data) {
            $('#nav-session-user').html('User: ' + data.name);
            userName.val("");
        });
    });

    $('#user-name').on('keypress', function(e) {
        if (e.which == 13) {
            $('#user-form-submit').click();
        }
    })


});
