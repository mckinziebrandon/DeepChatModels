// TODO: Add a username styling.
// TODO: Expanding / scrolling chat box.

$(document).ready(function() {
    // Extract the user input from the field.
    var user_msg = $('#chat-form :input[name="message"]');

    chat_submit = function(e) {
        var chatlog = $('#chat-log');
        chatlog.append("<div class='row'>\
            <div class='user-message text-right col-md-11'>" +
            user_msg.val() + "</div>" +
           "<div class='user-name text-right col-md-1'><b>User</b></div></div>"
        );

        // Submit a POST request to /chat
        $.post('chat/', {
            "message": user_msg.val()
        }, function(data) {
            chatlog.append("<div class='row'>\
                <div class='bot-name text-left col-md-1'><b>Botty</b></div>\
                <div class='bot-message text-left col-md-11'>" + data +
                "</div></div>"
            );
        });
    };

    $('#chat-submit').click(chat_submit);
    user_msg.keyup(function(code) {
        if (code.which == 13) {     // submit on ENTER key!
            chat_submit();
        }
    });
});
