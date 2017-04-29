$(document).ready(function() {

    // Extract the user input from the field.
    var user_msg = $('input[name="message"]');

    user_msg.on('keyup', function(e) {

        var dataName = $(this).attr('id');
        user_msg = $('#'+dataName);

        if (e.which != 13 || !user_msg.val()) {  // don't do anything on empty input
            return;
        }

        var chatLog = $('#'+dataName+'-chat-log');
        var messageRow = $('<div/>').attr('class', 'row message');

        messageRow.append($('<div>' + user_msg.val() + '</div>')
                .addClass('user-message text-left')
                .addClass('col-md-8 col-md-push-2')
                .addClass('col-sm-10 col-sm-push-2'));
        messageRow.append($('<div>User</div>')
                .addClass('user-name text-left')
                .addClass('col-md-2 col-md-offset-2'));
        messageRow.append($('<hr/>'));

        chatLog.append(messageRow);

        // Submit a POST request to collect user input.
        $.post('/', {
            "message": user_msg.val(),
            "dataName": dataName
        }, function(data) {
            $('#'+dataName+'-chat-log').append($('<div/>').addClass('row message')
                    .append($('<div/>')
                            .addClass('bot-name text-left col-md-2 col-sm-2')
                            .text('Botty'))
                    .append($('<div/>')
                            .addClass('bot-message text-left col-md-8 col-sm-9')
                            .text(data))
                    .append($('<hr/>')));
            chatLog.scrollTop(chatLog[0].scrollHeight);
            user_msg.val("");
        });
    });

});
