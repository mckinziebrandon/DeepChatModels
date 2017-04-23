// TODO: Add a username styling.
// TODO: Expanding / scrolling chat box.

$(document).ready(function() {
    // Extract the user input from the field.
    var user_msg = $('#chat-form :input[name="message"]');

    var chat_submit = function(e) {
        if (!user_msg.val()) {  // don't do anything on empty input
            return;
        }

        var chatlog = $('#chat-log');

        // Specify layout via class names (bootstrap).
        var messageCols = ''
        var userMsgCls = 'user-message text-left'
                + ' col-md-8 col-md-push-2'
                + ' col-sm-10 col-sm-push-2';

        // TODO: how does this look the way it looks...
        var userNameCls = 'user-name text-left'
                + ' col-md-2 col-md-offset-2';

        // User message - user name.
        var messageRow = $('<div/>').attr('class', 'row message');
        messageRow.append($('<div/>', {
           'class': userMsgCls,
            html: user_msg.val()}));
        messageRow.append($('<div/>', {
            'class': userNameCls,
            html: 'User'}));
        messageRow.append($('<hr/>'));
        chatlog.append(messageRow);

        // Submit a POST request to /chat
        $.post('/', {
            "message": user_msg.val()
        }, function(data) {
            chatlog.append("<div class='row message'>\
                <div class='bot-name text-left col-md-2 col-sm-2'>Botty</div>\
                <div class='bot-message text-left col-md-8 col-sm-9'>" + data +
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
