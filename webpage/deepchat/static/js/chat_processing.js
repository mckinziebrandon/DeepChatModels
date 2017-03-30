// TODO: Add a username styling.
// TODO: Expanding / scrolling chat box.

$(document).ready(function() {
    // Extract the user input from the field.
    var user_msg = $('#chat-form :input[name="message"]');

    chat_submit = function(e) {
        if (!user_msg.val()) {  // don't do anything on empty input
            return;
        }

        var chatlog = $('#chat-log');
        chatlog.append("<div class='row message'>\
            <div class='user-message text-left col-md-8 col-md-push-2'>" +
            user_msg.val() + "</div>" +
            "<div class='user-name text-left col-md-2 col-md-push-2'>\
            User</div></div><hr />"
        );

        // Submit a POST request to /chat
        $.post('chat/', {
            "message": user_msg.val()
        }, function(data) {
            chatlog.append("<div class='row message'>\
                <div class='bot-name text-left col-md-2'>Botty</div>\
                <div class='bot-message text-left col-md-8'>" + data +
                "</div></div><hr />"
            );
            $('#chat-log').scrollTop($('#chat-log')[0].scrollHeight);
            $('#chat-form :input[name="message"]').val("");
        });
    };

    $('#chat-submit').click(chat_submit);
    user_msg.keyup(function(code) {
        if (code.which == 13) {     // submit on ENTER key!
            chat_submit();
        }
    });
});
